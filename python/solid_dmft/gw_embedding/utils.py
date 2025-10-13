# -*- coding: utf-8 -*-
################################################################################
#
# solid_dmft - A versatile python wrapper to perform DFT+DMFT calculations
#              utilizing the TRIQS software library
#
# Copyright (C) 2018-2020, ETH Zurich
# Copyright (C) 2021, The Simons Foundation
#      authors: A. Hampel, M. Merkel, and S. Beck
#
# solid_dmft is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# solid_dmft is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# solid_dmft (in the file COPYING.txt in this directory). If not, see
# <http://www.gnu.org/licenses/>.
#
################################################################################
"""
utility functions for GW embedding
"""

import numpy as np
from scipy.constants import physical_constants

from h5 import HDFArchive
from triqs.utility import mpi
from triqs.gf import (
    Gf,
    BlockGf,
    make_gf_dlr,
)
from triqs.gf.meshes import MeshDLRImFreq
import itertools

from solid_dmft.gw_embedding.iaft import IAFT

HARTREE_EV = physical_constants['Hartree energy in eV'][0]

def get_dlr_from_IR(Gf_ir, ir_kernel, mesh_dlr_iw, dim=2):
    r"""
    Interpolate a given Gf from IR mesh to DLR mesh

    Parameters
    ----------
    Gf_ir : np.ndarray
        Green's function on IR mesh
    ir_kernel : sparse_ir
        IR kernel object
    mesh_dlr_iw : MeshDLRImFreq
        DLR mesh
    dim : int, optional
        dimension of the Green's function, defaults to 2

    Returns
    -------
    Gf_dlr : BlockGf or Gf
        Green's function on DLR mesh
    """

    n_orb = Gf_ir.shape[-1]
    stats = 'f' if mesh_dlr_iw.statistic == 'Fermion' else 'b'

    if stats == 'b':
        Gf_ir_pos = Gf_ir.copy()
        Gf_ir = np.zeros([Gf_ir_pos.shape[0] * 2 - 1] + [n_orb] * dim, dtype=complex)
        Gf_ir[: Gf_ir_pos.shape[0]] = Gf_ir_pos[::-1]
        Gf_ir[Gf_ir_pos.shape[0] :] = Gf_ir_pos[1:]

    Gf_dlr_iw = Gf(mesh=mesh_dlr_iw, target_shape=[n_orb] * dim)

    # prepare idx array for spare ir
    mesh_dlr_iw_idx = np.array([iwn.index for iwn in mesh_dlr_iw])

    Gf_dlr_iw.data[:] = ir_kernel.w_interpolate(Gf_ir, mesh_dlr_iw_idx, stats=stats, ir_notation=False)

    Gf_dlr = make_gf_dlr(Gf_dlr_iw)
    return Gf_dlr


def check_iaft_accuracy(Aw, ir_kernel, stats,
                        beta, dlr_wmax, dlr_prec, data_name):
    mpi.report('============== DLR mesh check ==============\n')
    mpi.report(f'Estimating accuracy of the user-defined (wmax, eps) = '
               f'({dlr_wmax}, {dlr_prec}) for the DLR mesh\n')
    ir_imp_kernel = IAFT(beta=beta, wmax=dlr_wmax, prec=dlr_prec)
    Aw_imp = ir_kernel.w_interpolate(Aw, ir_imp_kernel, 'f')

    ir_imp_kernel.check_leakage(Aw_imp, stats, data_name, w_input=True)
    mpi.report('=================== done ===================\n')


def estimate_zero_moment(Aw, iw_mesh):
    iw_m1 = iw_mesh[-1]
    iw_m2 = iw_mesh[-2]
    t = Aw[-1].real - (Aw[-1] - Aw[-2]).real * iw_m2 ** 2 / (
           iw_m2 ** 2 - iw_m1 ** 2)
    t = t.astype(complex)

    return t


def extract_h0_and_delta(g_weiss_wsIab, ir_kernel, high_freq_multiplier=10):
    """
    Estimate the static one-body term h₀ (as t_sIab) and the hybridization function Δ(iω)
    from a Weiss Green's function G₀(iω) sampled on a fermionic Matsubara mesh.

    The method:
      1) Interpolate G₀(iω) to a few very large Matsubara frequencies (scaled by
         `high_freq_multiplier`) to probe the asymptotic regime.
      2) Construct W(iω) = iω·I - [G₀(iω)]⁻¹ and estimate its zeroth moment
         t_sIab = lim_{|ω|→∞} W(iω) via `estimate_zero_moment`.
      3) Build Δ(iω) from the Dyson-like relation:
            Δ(iω) = iω·I - t_sIab - [G₀(iω)]⁻¹.

    Parameters
    ----------
    g_weiss_wsab : ndarray, complex, shape (nw, nspin, nbnd, nbnd)
        Weiss Green's function G₀(iωₙ) on the fermionic Matsubara mesh returned by `ir_kernel`.
        The leading dimension is frequency index; s,a,b are spin and orbital indices.
    ir_kernel : IAFT object
    high_freq_multiplier : float, default 10
        Multiplier applied to the last few (three) IR fermionic frequencies (in IR notation)
        before converting to physical Matsubara frequencies, to push evaluation deep into
        the asymptotic region for a more stable moment estimate.

    Returns
    -------
    t_sIab_estimate : ndarray, complex, shape (nspin, nbnd, nbnd)
        Estimate of the static one-body term (zeroth moment) per spin block.
    delta_estimate : ndarray, complex, shape (nw, nspin, nbnd, nbnd)
        Estimated hybridization function Δ(iωₙ) on the original fermionic mesh.

    Notes
    -----
    - Accuracy of `t_sIab_estimate` depends on how large the interpolated frequencies are.
    """
    nspin, nImp = g_weiss_wsIab.shape[1:3]

    # 1) Interpolate G0 to very high fermionic frequencies to improve the accuracy of high-frequency fitting
    iwn_interp = ir_kernel.wn_mesh('f', ir_notation=False)[-3:] * high_freq_multiplier
    g_weiss_interp = ir_kernel.w_interpolate(g_weiss_wsIab, iwn_interp, 'f', ir_notation=False)
    iwn_interp = (2 * iwn_interp.astype(float) + 1) * np.pi / ir_kernel.beta
    weiss_tmp = np.zeros(g_weiss_interp.shape, dtype=complex)
    for n, g in enumerate(g_weiss_interp):
        for s in range(nspin):
            for I in range(nImp):
                weiss_tmp[n,s,I] = 1j * iwn_interp[n] - np.linalg.inv(g[s,I])

    # 2) Fitting the zeroth moment as the non-interacting Hamiltonian
    t_sIab_estimate = estimate_zero_moment(weiss_tmp, iwn_interp)

    # 3) Construct Δ(iω) = iω·I - t_sIab - [G0(iω)]^{-1} on the original mesh
    iwn_mesh_imp = ir_kernel.wn_mesh('f') * np.pi / ir_kernel.beta
    delta_estimate = np.zeros(g_weiss_wsIab.shape, dtype=complex)
    nbnd = t_sIab_estimate.shape[-1]
    for n in range(delta_estimate.shape[0]):
        for s in range(nspin):
            for I in range(nImp):
                g_weiss_inv = np.linalg.inv(g_weiss_wsIab[n,s,I])
                delta_estimate[n,s,I] = 1j * iwn_mesh_imp[n] * np.eye(nbnd) - t_sIab_estimate[s,I] - g_weiss_inv

    # 4) checking the leakage of the resulting Δ(iω)
    ir_kernel.check_leakage(delta_estimate, 'f', 'delta_estimate', w_input=True)

    return t_sIab_estimate, delta_estimate


def read_t_and_delta(aimb_h5, it_1e=None):
    mpi.report(f"Reading the analytic H_loc0 and the corresponding hybridization from aimbes h5 {aimb_h5}.")
    with HDFArchive(aimb_h5, 'r') as ar:
        if not it_1e:
            it_1e = ar['downfold_1e/final_iter']
        iter_grp = ar[f'downfold_1e/iter{it_1e}']

        t_sIab = iter_grp['H0_sIab'] + iter_grp['Vhf_gw_sIab'] - iter_grp['Vhf_dc_sIab']
        if 'Vcorr_gw_sIab' in iter_grp:
            t_sIab += (iter_grp['Vcorr_gw_sIab'] - iter_grp['Vcorr_dc_sIab'])

        nspin, nImp, nOrbs = t_sIab.shape[:3]
        for s in np.arange(nspin):
            for I in range(nImp):
                t_sIab[s, I] -= np.eye(nOrbs) * iter_grp["mu"]

        delta_wsIab = iter_grp['delta_wsIab']

    return t_sIab, delta_wsIab


def get_ir_imp_kernel(ir_kernel, gw_data):
    # if user-defined wmax and eps for the impurity
    # Here we only change wmax while keeping prec the same
    custom_ir_kernel = ir_kernel.wmax != gw_data["imp_wmax"]
    if custom_ir_kernel:
        ir_imp_kernel = IAFT(
            beta=gw_data['beta'],
            wmax=gw_data['imp_wmax'],
            prec=gw_data['gw_prec'],
            verbose=True
        )
        return ir_imp_kernel
    else:
        return ir_kernel


def compute_g_weiss(t_sIab, delta_wsIab, ir_kernel):
    # reconstruct g_weiss
    g_weiss = np.zeros(delta_wsIab.shape, dtype=complex)
    eye = np.eye(delta_wsIab.shape[-1], dtype=complex)
    iw_mesh = 1j * ir_kernel.wn_mesh('f').astype(float) * np.pi / ir_kernel.beta
    for n, iw in enumerate(iw_mesh):
        for s in range(delta_wsIab.shape[1]):
            for I in range(delta_wsIab.shape[2]):
                # g_weiss(iw) = [ iw - t - delta(iw) ]^-1
                tmp = iw * eye - t_sIab[s, I] - delta_wsIab[n, s, I]
                g_weiss[n, s, I] = np.linalg.inv(tmp)
    return g_weiss


def tail_fit_g_weiss(g_weiss_wsIab, ir_kernel, high_freq_multiplier=10):
    mpi.report("Extracting H_loc0 and hybridization from tail fitting fermionic Weiss field g.\n")
    # extracting H0 and Delta from g_weiss
    t_sIab, delta_wsIab = extract_h0_and_delta(g_weiss_wsIab, ir_kernel, high_freq_multiplier)

    return compute_g_weiss(t_sIab, delta_wsIab, ir_kernel), t_sIab, delta_wsIab


def causal_projection(A_wsab, iw_mesh, statistics, Np=5,
                      *, name="", iw_mesh_out=None):
    mpi.report(f"Causal projection for {name} with nbath/orbital = {Np} and statistics={statistics}")
    try:
        from adapol import hybfit
    except ImportError:
        raise ImportError("bath fitting requires the adapol package (https://github.com/flatironinstitute/adapol). \n"
                          "Please ensure that it is installed. ")
    if statistics not in ("fermion", "boson"):
        raise ValueError(f"Invalid statistics: {statistics!r}. Use 'fermion' or 'boson'.")
    if A_wsab.ndim < 3:
        raise ValueError("A_wsab must be at least 3D: (Nw, norb, norb) or (Nw, nspin, norb, norb).")
    if A_wsab.shape[-1] != A_wsab.shape[-2]:
        raise ValueError("The last two dimensions of A_wsab must be equal (square matrices).")
    if A_wsab.shape[0] != iw_mesh.shape[0]:
        raise ValueError("Mismatch: A_wsab.shape[0] (Nw) != iw_mesh.shape[0].")

    if iw_mesh_out is None:
        iw_mesh_out = iw_mesh

    original_shape = None
    if len(A_wsab.shape) != 4:
        original_shape = A_wsab.shape
        A_wsab = A_wsab.reshape(A_wsab.shape[0], -1, A_wsab.shape[-2], A_wsab.shape[-1])

    nspin, error = A_wsab.shape[1], -1
    A_out = np.zeros((iw_mesh_out.shape[0],) + A_wsab.shape[1:], dtype=A_wsab.dtype)
    for s in np.arange(nspin):
        _, __, fit_error, func = hybfit(
            A_wsab[:, s], iw_mesh, Np=Np, solver='sdp', verbose=False, statistics=statistics
        )
        A_out[:, s] = func(iw_mesh_out)
        error = max(error, abs(fit_error))
    mpi.report(f"Causal projection error =  {error}")

    if original_shape is not None:
        A_wsab = A_wsab.reshape(original_shape)
        A_out = A_out.reshape((iw_mesh_out.shape[0],) + original_shape[1:])

    return A_out
