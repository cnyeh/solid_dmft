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
converter from bdft output to edmft input for solid_dmft
"""

import numpy as np
from scipy.constants import physical_constants


from h5 import HDFArchive
from triqs.utility import mpi
from triqs.gf import (
    Gf,
    BlockGf,
    make_gf_dlr_imtime,
    make_gf_dlr,
    make_gf_dlr_imfreq,
)
from triqs.gf.meshes import MeshDLRImFreq, MeshDLRImTime
import itertools

from solid_dmft.gw_embedding.utils import get_dlr_from_IR, tail_fit_g_weiss, causal_projection
from solid_dmft.gw_embedding.iaft import IAFT, set_precision

HARTREE_EV = physical_constants['Hartree energy in eV'][0]


def convert_gw_output(job_h5, gw_h5, dlr_wmax=None, dlr_eps=None,
                      it_1e=0, it_2e=0,
                      delta_calc_type="tail_fit", delta_causal_fit=False,
                      u_zero_slope=False,
                      ha_ev_conv = False):
    """
    read bdft output and convert to triqs Gf DLR objects

    Parameters
    ----------
    job_h5: string
        path to solid_dmft job file
    gw_h5: string
        path to GW checkpoint file for AIMBES code
    dlr_wmax: float
        wmax for dlr mesh, defaults to the wmax from the IR basis
    dlr_eps: float
        precision for dlr mesh, defaults to the precision from the IR basis
    it_1e: int, optional
        iteration to read from gw_h5 calculation for 1e downfolding, defaults to last iteration
    it_2e: int, optional
        iteration to read from gw_h5 calculation for 2e downfolding, defaults to last iteration
    ha_ev_conv: bool, optional
        convert energies from Hartree to eV, defaults to False

    Returns
    -------
    gw_data: dict
        dictionary holding all read objects: mu_emb, beta, lam, w_max, prec, mesh_dlr_iw_b,
        mesh_dlr_iw_f, n_orb, G0_dlr, Gloc_dlr, Sigma_imp_dlr, Sigma_imp_DC_dlr, Uloc_dlr,
        Vloc, Hloc0, Vhf_dc, Vhf
    ir_kernel: sparse_ir kernel object
        IR kernel with AIMBES paramaters
    """

    gw_data = {}

    if ha_ev_conv:
        conv_fac = HARTREE_EV
    else:
        conv_fac = 1.0

    with HDFArchive(gw_h5, 'r') as ar:
        if not it_1e or not it_2e:
            it_1e = ar['downfold_1e/final_iter']
            it_2e = ar['downfold_2e/final_iter']

        mpi.report(f'Reading results from downfold_1e iter {it_1e} and downfold_2e iter {it_2e} from the CoQui checkpoint.')

        # auxilary quantities
        gw_data['it_1e'] = it_1e
        gw_data['it_2e'] = it_2e
        gw_data['mu_emb'] = ar[f'downfold_1e/iter{it_1e}']['mu']
        gw_data['beta'] = ar['imaginary_fourier_transform']['beta']
        gw_data['lam'] = ar['imaginary_fourier_transform']['lambda']
        if 'wmax' in ar['imaginary_fourier_transform']:
            gw_data['gw_wmax'] = ar['imaginary_fourier_transform']['wmax']
        else:
            gw_data['gw_wmax'] = gw_data['lam'] / gw_data['beta']
        gw_data['gw_dlr_wmax'] = gw_data['gw_wmax'] if dlr_wmax is None else dlr_wmax
        gw_data['number_of_spins'] = ar['system/number_of_spins']
        assert gw_data['number_of_spins'] == 1, 'spin calculations not yet supported in converter'

        prec = ar['imaginary_fourier_transform']['prec']
        gw_data['gw_ir_prec'] = set_precision(prec)
        if dlr_eps is None:
            gw_data['gw_ir_prec'] = gw_data['gw_ir_prec'] if gw_data['gw_ir_prec'] >= 1e-13 else 1e-13
        else:
            gw_data['gw_dlr_prec'] = dlr_eps

        # 1 particle properties
        g_weiss_wsIab = ar[f'downfold_1e/iter{it_1e}']['g_weiss_wsIab']
        delta_wsIab = np.zeros(g_weiss_wsIab.shape, dtype=g_weiss_wsIab.dtype)
        Sigma_dc_wsIab = ar[f'downfold_1e/iter{it_1e}']['Sigma_dc_wsIab']
        Gloc = ar[f'downfold_1e/iter{it_1e}']['Gloc_wsIab']
        gw_data['n_inequiv_shells'] = Gloc.shape[2]

        # 2 particle properties
        # TODO: discuss how the site index is used right now in bDFT
        Vloc_jk = ar[f'downfold_2e/iter{it_2e}']['Vloc_abcd']
        Uloc_ir_jk = ar[f'downfold_2e/iter{it_2e}']['Uloc_wabcd'][:, ...]
        # switch inner two indices to match triqs notation
        Vloc = np.zeros(Vloc_jk.shape, dtype=complex)
        Uloc_ir = np.zeros(Uloc_ir_jk.shape, dtype=complex)
        n_orb = Vloc.shape[0]
        for or1, or2, or3, or4 in itertools.product(range(n_orb), repeat=4):
            Vloc[or1, or2, or3, or4] = Vloc_jk[or1, or3, or2, or4]
            for ir_w in range(Uloc_ir_jk.shape[0]):
                Uloc_ir[ir_w, or1, or2, or3, or4] = Uloc_ir_jk[ir_w, or1, or3, or2, or4]

        if u_zero_slope:
            mpi.report("Enforce U(iw=0) equals to its closest neighbor for numerical stability."
                       "Once the EDMFT loop converges, set \"u_zero_slope=false\" for further convergence.")
            Uloc_ir[0] = Uloc_ir[1]

        if 'Pi_dc_wabcd' in ar[f'downfold_2e/iter{it_2e}']:
            Pi_DC_ir = ar[f'downfold_2e/iter{it_2e}']['Pi_dc_wabcd']
        else:
            Pi_DC_ir = np.zeros(Uloc_ir.shape)

        Vhf_dc_sIab = ar[f'downfold_1e/iter{it_1e}']['Vhf_dc_sIab'][0, 0]
        if 'Vhf_gw_sIab' in ar[f'downfold_1e/iter{it_1e}']:
            Vhf_sIab = ar[f'downfold_1e/iter{it_1e}']['Vhf_gw_sIab'][0, 0]
        else:
            Vhf_sIab = np.zeros(Vhf_dc_sIab.shape, dtype=complex)

        if 'Vcorr_gw_sIab' in ar[f'downfold_1e/iter{it_1e}']:
            mpi.report('Found Vcorr_sIab in the bdft checkpoint file, '
                       'i.e. Embedding on top of an effective QP Hamiltonian.')
            qp_emb = True
        else:
            qp_emb = False
        mpi.report("")

    # get IR object
    mpi.report('Creating IR kernel and convert to DLR.')
    # create IR kernel
    mpi.report("\nReading IR representation from CoQuí...")
    ir_kernel = IAFT(beta=gw_data['beta'], wmax=gw_data['gw_wmax'], prec=gw_data['gw_ir_prec'])

    mpi.report("Constructing DLR mesh (wmax, eps) = ({}, {})...".format(gw_data['gw_dlr_wmax'], gw_data['gw_dlr_prec']))
    gw_data['mesh_dlr_iw_b'] = MeshDLRImFreq(
        beta=gw_data['beta'] / conv_fac,
        statistic='Boson',
        w_max=gw_data['gw_dlr_wmax'] * conv_fac,
        eps=gw_data['gw_dlr_prec'],
        symmetrize=True
    )
    gw_data['mesh_dlr_iw_f'] = MeshDLRImFreq(
        beta=gw_data['beta'] / conv_fac,
        statistic='Fermion',
        w_max=gw_data['gw_dlr_wmax'] * conv_fac,
        eps=gw_data['gw_dlr_prec'],
        symmetrize=True
    )

    if delta_calc_type not in {"analytic", "tail_fit"}:
        raise ValueError("calc_type must be either \'analytic\' or \'tail_fit\'.")

    if delta_calc_type == "analytic":
        raise ValueError("delta_calc_type = analytic is deprecated. "
                         "Please set delta_calc_type == tail_fit.")
        Hloc0, delta_wsIab = None, None
        ir_imp_kernel = ir_kernel
    elif delta_calc_type == "tail_fit":
        Hloc0, delta_wsIab, ir_imp_kernel = tail_fit_g_weiss(g_weiss_wsIab, ir_kernel, gw_data,
                                                             wmax_imp=dlr_wmax, eps_imp=dlr_eps)
    if delta_causal_fit:
        delta_wsIab = causal_projection(
            delta_wsIab, ir_imp_kernel.wn_mesh('f')*np.pi/ir_imp_kernel.beta,
            statistics="fermion", name="hybridization",
            Np=8
        )

    Hloc0 = Hloc0[0,0]

    # need to update g_weiss?

    mpi.report("")

    (
        U_dlr_list,
        Pi_DC_dlr_list,
        G0_dlr_list,
        delta_dlr_list,
        Gloc_dlr_list,
        Sigma_dlr_list,
        Sigma_DC_dlr_list,
        V_list,
        Hloc_list,
        Vhf_list,
        Vhf_dc_list,
        n_orb_list,
    ) = [], [], [], [], [], [], [], [], [], [], [], []
    for ish in range(gw_data['n_inequiv_shells']):
        # fit IR Uloc on DLR iw mesh
        temp = get_dlr_from_IR(Uloc_ir*conv_fac, ir_kernel, gw_data['mesh_dlr_iw_b'], dim=4)
        Uloc_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
        U_dlr_list.append(Uloc_dlr)
        # in product basis
        temp = get_dlr_from_IR(Pi_DC_ir.reshape(-1, n_orb**2, n_orb**2)*conv_fac, ir_kernel, gw_data['mesh_dlr_iw_b'], dim=2)
        Pi_DC_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
        Pi_DC_dlr_list.append(Pi_DC_dlr)
        V_list.append({'up': Vloc.copy()*conv_fac, 'down': Vloc*conv_fac})
        Hloc_list.append({'up': Hloc0.copy()*conv_fac, 'down': Hloc0*conv_fac})
        Vhf_list.append({'up': Vhf_sIab.copy()*conv_fac, 'down': Vhf_sIab*conv_fac})
        Vhf_dc_list.append({'up': Vhf_dc_sIab.copy()*conv_fac, 'down': Vhf_dc_sIab*conv_fac})
        n_orb_list.append(n_orb)

        temp = get_dlr_from_IR(g_weiss_wsIab[:, 0, ish, :, :]/conv_fac, ir_kernel, gw_data['mesh_dlr_iw_f'], dim=2)
        G0_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
        G0_dlr_list.append(G0_dlr)

        # FIXME make consistent usage of ir_kernel and ir_imp_kernel
        temp = get_dlr_from_IR(delta_wsIab[:, 0, ish, :, :]/conv_fac, ir_imp_kernel, gw_data['mesh_dlr_iw_f'], dim=2)
        delta_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
        delta_dlr_list.append(delta_dlr)

        temp = get_dlr_from_IR(Gloc[:, 0, ish, :, :]/conv_fac, ir_kernel, gw_data['mesh_dlr_iw_f'], dim=2)
        Gloc_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
        Gloc_dlr_list.append(Gloc_dlr)

        temp = get_dlr_from_IR(Sigma_dc_wsIab[:, 0, ish, :, :]*conv_fac, ir_kernel, gw_data['mesh_dlr_iw_f'], dim=2)
        Sigma_DC_dlr = BlockGf(name_list=['up', 'down'], block_list=[temp, temp], make_copies=True)
        Sigma_DC_dlr_list.append(Sigma_DC_dlr)

    gw_data['G0_dlr'] = G0_dlr_list
    gw_data['delta_dlr'] = delta_dlr_list
    gw_data['Gloc_dlr'] = Gloc_dlr_list
    gw_data['Sigma_imp_DC_dlr'] = Sigma_DC_dlr_list
    gw_data['Uloc_dlr'] = U_dlr_list
    gw_data['Pi_DC_dlr'] = Pi_DC_dlr_list
    gw_data['Vloc'] = V_list
    gw_data['Hloc0'] = Hloc_list
    gw_data['Vhf_dc'] = Vhf_dc_list
    gw_data['Vhf'] = Vhf_list
    gw_data['n_orb'] = n_orb_list

    # write Uloc / Wloc back to h5 archive
    mpi.report(f'Writing results in {job_h5}/DMFT_input')

    with HDFArchive(job_h5, 'a') as ar:
        if 'DMFT_input' not in ar:
            ar.create_group('DMFT_input')
        if f'iter{it_1e}' not in ar['DMFT_input']:
            ar['DMFT_input'].create_group(f'iter{it_1e}')

        for key, value in gw_data.items():
            ar[f'DMFT_input/iter{it_1e}'][key] = value

    mpi.report(f'finished writing results in {job_h5}/DMFT_input')
    return gw_data, ir_kernel


