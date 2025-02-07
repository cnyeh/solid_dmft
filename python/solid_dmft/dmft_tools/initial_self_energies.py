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
Contains all functions related to determining the double counting and the
initial self-energy.
"""

# system
from copy import deepcopy
import numpy as np

# triqs
from h5 import HDFArchive
import triqs.utility.mpi as mpi
from triqs.gf import BlockGf, Gf, make_gf_imfreq, MeshDLRImFreq, make_gf_dlr, MeshReFreq
import itertools

def calculate_double_counting(sum_k, density_matrix, general_params, gw_params, advanced_params, solver_type_per_imp, G_loc_all=None):
    """
    Calculates the double counting, including all manipulations from advanced_params.

    Parameters
    ----------
    sum_k : SumkDFT object
    density_matrix : list of gf_struct_solver like
        List of density matrices for all inequivalent shells
    general_params : dict
        general parameters as a dict
    gw_params : dict
        GW parameters as a dict
    advanced_params : dict
        advanced parameters as a dict
    solver_type_per_imp : list of str
        List of solver types for each impurity
    G_loc_all : list of BlockGf (Green's function) objects, optional
        List of local Green's functions for all shells

    Returns
    --------
    sum_k : SumKDFT object
        The SumKDFT object containing the updated double counting
    """

    mpi.report('\n*** DC determination ***')

    # copy the density matrix to not change it
    density_matrix_DC = deepcopy(density_matrix)

    # TODO: suppress print when reseting DC to zero
    #       and add a final print of the DC pot/energy at the end of the whole function
    icrsh_hartree = [icrsh for icrsh, type in enumerate(solver_type_per_imp) if type == 'hartree']
    icrsh_not_hartree = [icrsh for icrsh, type in enumerate(solver_type_per_imp) if type != 'hartree']
    if icrsh_hartree:
        mpi.report(f'\nSOLID_DMFT: Hartree solver for impurities {icrsh_hartree} detected. '
                   'Zeroing out the DC correction there, which gets computed at the solver level.')
        for icrsh in icrsh_hartree:
            sum_k.calc_dc(density_matrix_DC[icrsh], orb=icrsh,
                          use_dc_value=0.0)

    # Sets the DC and exits the function if advanced_params['dc_fixed_value'] is specified
    if advanced_params['dc_fixed_value'] is not None:
        for icrsh in icrsh_not_hartree:
            sum_k.calc_dc(density_matrix_DC[icrsh], orb=icrsh,
                          use_dc_value=advanced_params['dc_fixed_value'])
        return sum_k

    for icrsh in icrsh_not_hartree:
        if advanced_params['dc_fixed_occ'][icrsh] is not None:
            mpi.report(f'Fixing occupation for DC for imp {icrsh} to n={advanced_params["dc_fixed_occ"][icrsh]:.4f}')
            n_orb = sum_k.corr_shells[icrsh]['dim']
            # we need to handover a matrix to calc_dc so calc occ per orb per spin channel
            orb_occ = advanced_params['dc_fixed_occ'][icrsh]/(n_orb*2)
            # setting occ of each diag orb element to calc value
            for inner in density_matrix_DC[icrsh].values():
                np.fill_diagonal(inner, orb_occ+0.0j)

    # The regular way: calculates the DC based on U, J and the dc_type
    for icrsh in icrsh_not_hartree:
        if general_params['dc_type'][icrsh] == 3:
            # this is FLL for eg orbitals only as done in Seth PRB 96 205139 2017 eq 10
            # this setting for U and J is reasonable as it is in the spirit of F0 and Javg
            # for the 5 orb case
            mpi.report('Doing FLL DC for eg orbitals only with Uavg=U-J and Javg=2*J')
            Uavg = advanced_params['dc_U'][icrsh] - advanced_params['dc_J'][icrsh]
            Javg = 2*advanced_params['dc_J'][icrsh]
            sum_k.calc_dc(density_matrix_DC[icrsh], U_interact=Uavg, J_hund=Javg,
                          orb=icrsh, use_dc_formula=0)
        # DC calculated for dynamic interaction from AIMBES
        elif general_params['dc_type'][icrsh] in ('crpa_static', 'crpa_static_qp', 'crpa_dynamic'):
            from solid_dmft.gw_embedding.bdft_converter import calc_Sigma_DC_gw, calc_W_from_Gloc, convert_gw_output
            mpi.report('\n*** Using dynamic interactions to calculate DC ***')

            # lad GW input from h5 file
            if 'Uloc_dlr' not in gw_params:
                if mpi.is_master_node():
                    gw_data, ir_kernel = convert_gw_output(
                        general_params['jobname'] + '/' + general_params['seedname'] + '.h5',
                        gw_params['h5_file'],
                        it_1e = gw_params['it_1e'],
                        it_2e = gw_params['it_2e'],
                        ha_ev_conv = True
                    )
                    gw_params.update(gw_data)
                gw_params = mpi.bcast(gw_params)

            mesh = MeshDLRImFreq(sum_k.mesh.beta, 'Fermion',
                                 sum_k.mesh(sum_k.mesh.last_index()).value.imag, gw_params['Uloc_dlr'][icrsh].mesh.eps)
            Gloc_dlr_iw = sum_k.block_structure.create_gf(ish=icrsh, space='sumk', mesh=mesh)

            G_loc_sumk = sum_k.block_structure.convert_gf(G_loc_all[icrsh], ish_from=icrsh, space_from='solver', space_to='sumk')
            for block, gf in Gloc_dlr_iw:
                for iw in gf.mesh:
                    gf[iw] = G_loc_sumk[block](iw)
            Gloc_dlr = make_gf_dlr(Gloc_dlr_iw)

            U_matrix_rot = {'up' : gw_params['U_matrix_rot'][icrsh], 'down' : gw_params['U_matrix_rot'][icrsh]}

            # there are two options here evaluate DC from Wloc_GW and Uloc
            # or Wloc_GG and Uloc (here GG means Wloc calculated via Gloc*Gloc)
            Wloc_dlr = calc_W_from_Gloc(Gloc_dlr, U_matrix_rot)
            Sig_DC_dlr, Sig_DC_hartree, Sig_DC_exchange = calc_Sigma_DC_gw(Wloc_dlr,
                                                                           Gloc_dlr,
                                                                           U_matrix_rot)
            Sig_DC_iw = make_gf_imfreq(Sig_DC_dlr, n_iw=len(sum_k.mesh)//2)
            Sig_DC_iw_dyn = Sig_DC_iw.copy()
            for block, gf in Sig_DC_iw:
                for iorb, jorb in itertools.product(range(gf.target_shape[0]), repeat=2):
                    # create full freq dependent DC
                    gf[iorb, jorb] += Sig_DC_hartree[block][iorb, jorb].real + Sig_DC_exchange[block][iorb, jorb].real

            # dynamic interaction but static DC
            if general_params['dc_type'][icrsh] == 'crpa_static':
                # for the static DC form we follow doi.org/10.1103/PhysRevB.95.155104 Eq 31
                # Sig_DC = Sig_DC_hartree + Sig_DC_exchange
                for block, gf in Sig_DC_iw:
                    Sig_DC_hartree[block] += Sig_DC_exchange[block]

                    mpi.report(f'DC for imp {icrsh} block {block} via Σ_dc_HF + Σ_dc_ex:')
                    mpi.report(Sig_DC_hartree[block].real)
                # transform dc to sumk blocks
                sum_k.dc_imp[icrsh] = Sig_DC_hartree

            elif general_params['dc_type'][icrsh] == 'crpa_static_qp':
                # for the static DC on top of GW we follow doi.org/10.1103/PhysRevB.95.155104 Eq 31
                # Sig_DC = Sig_DC_hartree + Sig_DC_exchange + Sig_DC_iw(0)
                mesh_w = MeshReFreq(window=(-0.5,0.5), n_w=101)
                Sig_DC_w = sum_k.block_structure.create_gf(ish=icrsh, space='sumk', mesh=mesh_w)
                for block, gf in Sig_DC_w:
                    gf.set_from_pade(Sig_DC_iw[block], n_points=len(sum_k.mesh)//10, freq_offset=0.0001)
                    Sig_DC_hartree[block] = 0.5*(Sig_DC_w[block](0.0) + Sig_DC_w[block](0.0).conj().T).real
                    mpi.report(f'DC for imp {icrsh} block {block} via Σ_dc_HF + Σ_dc_ex:')
                    mpi.report(Sig_DC_hartree[block].real)
                # transform dc to sumk blocks
                sum_k.dc_imp[icrsh] = Sig_DC_hartree

            elif general_params['dc_type'][icrsh] == 'crpa_dynamic':
                for block, gf in Sig_DC_iw:
                    mpi.report(f'Full dynamic DC from cRPA for imp {icrsh} block {block} at iw_n=0:')
                    mpi.report(gf(0).real)
                    mpi.report(f'Full dynamic DC from cRPA for imp {icrsh} block {block} at iw_n=n:')
                    mpi.report(gf.data[-1,:,:].real)
                    Sig_DC_hartree[block] += Sig_DC_exchange[block]

                # sum_k.dc_imp stores the sumk block structure version
                sum_k.dc_imp[icrsh] = Sig_DC_hartree

                # the dynamic part of DC is stored in different object
                sum_k.dc_imp_dyn[icrsh] = Sig_DC_iw_dyn

        else:
            mpi.report(f'\nCalculating standard DC for impurity {icrsh} with U={advanced_params["dc_U"][icrsh]} and J={advanced_params["dc_J"][icrsh]}')
            sum_k.calc_dc(density_matrix_DC[icrsh], U_interact=advanced_params['dc_U'][icrsh],
                          J_hund=advanced_params['dc_J'][icrsh], orb=icrsh,
                          use_dc_formula=general_params['dc_type'][icrsh])

    # for the fixed DC according to https://doi.org/10.1103/PhysRevB.90.075136
    # dc_imp is calculated with fixed occ but dc_energ is calculated with given n
    if advanced_params['dc_nominal']:
        if 'Hartree' in solver_type_per_imp:
            raise NotImplementedError('dc_nominal not implemented in presence of Hartree solver')
        mpi.report('\ncalculating DC energy with fixed DC potential from above\n'
                   + ' for the original density matrix doi.org/10.1103/PhysRevB.90.075136\n'
                   + ' aka nominal DC')
        dc_imp = deepcopy(sum_k.dc_imp)
        dc_new_en = deepcopy(sum_k.dc_energ)
        for ish in range(sum_k.n_corr_shells):
            n_DC = 0.0
            for value in density_matrix[sum_k.corr_to_inequiv[ish]].values():
                n_DC += np.trace(value.real)

            # calculate new DC_energ as n*V_DC
            # average over blocks in case blocks have different imp
            dc_new_en[ish] = 0.0
            for spin, dc_per_spin in dc_imp[ish].items():
                # assuming that the DC potential is the same for all orbitals
                # dc_per_spin is a list for each block containing on the diag
                # elements the DC potential for the self-energy correction
                dc_new_en[ish] += n_DC * dc_per_spin[0][0]
            dc_new_en[ish] = dc_new_en[ish] / len(dc_imp[ish])
        sum_k.set_dc(dc_imp, dc_new_en)

        # Print new DC values
        mpi.report('\nFixed occ, new DC values:')
        for icrsh, (dc_per_shell, energy_per_shell) in enumerate(zip(dc_imp, dc_new_en)):
            for spin, dc_per_spin in dc_per_shell.items():
                mpi.report('DC for shell {} and block {} = {}'.format(icrsh, spin, dc_per_spin[0][0]))
            mpi.report('DC energy for shell {} = {}'.format(icrsh, energy_per_shell))

    # Rescales DC if advanced_params['dc_factor'] is given
    if advanced_params['dc_factor'] is not None:
        # Here, no check for Hartree since its DC is 0 and the scaling doesn't change that
        rescaled_dc_imp = [{spin: advanced_params['dc_factor'] * dc_per_spin
                            for spin, dc_per_spin in dc_per_shell.items()}
                          for dc_per_shell in sum_k.dc_imp]
        rescaled_dc_energy = [advanced_params['dc_factor'] * energy_per_shell
                              for energy_per_shell in sum_k.dc_energ]
        sum_k.set_dc(rescaled_dc_imp, rescaled_dc_energy)

        # Print new DC values
        mpi.report('\nRescaled DC, new DC values:')
        for icrsh, (dc_per_shell, energy_per_shell) in enumerate(zip(rescaled_dc_imp, rescaled_dc_energy)):
            for spin, dc_per_spin in dc_per_shell.items():
                mpi.report('DC for shell {} and block {} = {}'.format(icrsh, spin, dc_per_spin[0][0]))
            mpi.report('DC energy for shell {} = {}'.format(icrsh, energy_per_shell))

    if advanced_params['dc_orb_shift'] is not None:
        if 'Hartree' in solver_type_per_imp:
            raise NotImplementedError('dc_orb_shift not implemented in presence of Hartree solver')
        mpi.report('adding an extra orbital dependent shift per impurity')
        tot_norb = 0
        dc_orb_shift = []
        dc_orb_shift_orig = deepcopy(advanced_params['dc_orb_shift'])
        for icrsh in range(sum_k.n_inequiv_shells):
            tot_norb += sum_k.corr_shells[icrsh]['dim']
            dc_orb_shift.append(dc_orb_shift_orig[:sum_k.corr_shells[icrsh]['dim']])
            del dc_orb_shift_orig[:sum_k.corr_shells[icrsh]['dim']]

        dc_orb_shift = np.array(dc_orb_shift)
        dc = []
        for icrsh in range(sum_k.n_inequiv_shells):
            mpi.report(f'shift on imp {icrsh}: {dc_orb_shift[icrsh,:]}')
            dc.append({})
            for spin, dc_per_spin in sum_k.dc_imp[sum_k.inequiv_to_corr[icrsh]].items():
                dc[icrsh][spin] = dc_per_spin + np.diag(dc_orb_shift[icrsh,:])

        for ish in range(sum_k.n_corr_shells):
            sum_k.dc_imp[ish] = dc[sum_k.corr_to_inequiv[ish]]

    return sum_k


def _load_sigma_from_h5(h5_archive, iteration):
    """
    Reads impurity self-energy for all impurities from file and returns them as a list

    Parameters
    ----------
    h5_archive : HDFArchive
        HDFArchive to read from
    iteration : int
        at which iteration will sigma be loaded

    Returns
    --------
    self_energies : list of green functions

    dc_imp : numpy array
        DC potentials
    dc_energy : numpy array
        DC energies per impurity
    density_matrix : numpy arrays
        Density matrix from the previous self-energy
    """

    internal_path = 'DMFT_results/'
    internal_path += 'last_iter' if iteration == -1 else 'it_{}'.format(iteration)

    n_inequiv_shells = h5_archive['dft_input']['n_inequiv_shells']

    # Loads previous self-energies and DC
    self_energies = [h5_archive[internal_path]['Sigma_freq_{}'.format(iineq)]
                     for iineq in range(n_inequiv_shells)]
    last_g0 = [h5_archive[internal_path]['G0_freq_{}'.format(iineq)]
                     for iineq in range(n_inequiv_shells)]
    dc_imp = h5_archive[internal_path]['DC_pot']
    dc_energy = h5_archive[internal_path]['DC_energ']

    # Loads density_matrix to recalculate DC if dc_dmft
    density_matrix = h5_archive[internal_path]['dens_mat_post']

    print('Loaded Sigma_imp0...imp{} '.format(n_inequiv_shells-1)
          + ('at last it ' if iteration == -1 else 'at it {} '.format(iteration)))

    return self_energies, dc_imp, dc_energy, last_g0, density_matrix


def _sumk_sigma_to_solver_struct(sum_k, start_sigma):
    """
    Extracts the local Sigma. Copied from SumkDFT.extract_G_loc, version 2.1.x.

    Parameters
    ----------
    sum_k : SumkDFT object
        Sumk object with the information about the correct block structure
    start_sigma : list of BlockGf (Green's function) objects
        List of Sigmas in sum_k block structure that are to be converted.

    Returns
    -------
    Sigma_inequiv : list of BlockGf (Green's function) objects
        List of Sigmas that can be used to initialize the solver
    """

    Sigma_local = [start_sigma[icrsh].copy() for icrsh in range(sum_k.n_corr_shells)]
    Sigma_inequiv = [BlockGf(name_block_generator=[(block, Gf(mesh=Sigma_local[0].mesh, target_shape=(dim, dim)))
                                                   for block, dim in sum_k.gf_struct_solver[ish].items()],
                             make_copies=False) for ish in range(sum_k.n_inequiv_shells)]

    # G_loc is rotated to the local coordinate system
    if sum_k.use_rotations:
        for icrsh in range(sum_k.n_corr_shells):
            for bname, gf in Sigma_local[icrsh]:
                Sigma_local[icrsh][bname] << sum_k.rotloc(
                    icrsh, gf, direction='toLocal')

    # transform to CTQMC blocks
    for ish in range(sum_k.n_inequiv_shells):
        for block, dim in sum_k.gf_struct_solver[ish].items():
            for ind1 in range(dim):
                for ind2 in range(dim):
                    block_sumk, ind1_sumk = sum_k.solver_to_sumk[ish][(block, ind1)]
                    block_sumk, ind2_sumk = sum_k.solver_to_sumk[ish][(block, ind2)]
                    Sigma_inequiv[ish][block][ind1, ind2] << Sigma_local[
                        sum_k.inequiv_to_corr[ish]][block_sumk][ind1_sumk, ind2_sumk]

    # return only the inequivalent shells
    return Sigma_inequiv


def _set_loaded_sigma(sum_k, loaded_sigma, loaded_dc_imp, general_params):
    """
    Adjusts for the Hartree shift when loading a self energy Sigma_freq from a
    previous calculation that was run with a different U, J or double counting.

    Parameters
    ----------
    sum_k : SumkDFT object
        Sumk object with the information about the correct block structure
    loaded_sigma : list of BlockGf (Green's function) objects
        List of Sigmas loaded from the previous calculation
    loaded_dc_imp : list of dicts
        List of dicts containing the loaded DC. Used to adjust the Hartree shift.
    general_params : dict
        general parameters as a dict

    Raises
    ------
    ValueError
        Raised if the block structure between the loaded and the Sumk DC_imp
        does not agree.

    Returns
    -------
    start_sigma : list of BlockGf (Green's function) objects
        List of Sigmas, loaded Sigma adjusted for the new Hartree term

    """
    # Compares loaded and new double counting
    if len(loaded_dc_imp) != len(sum_k.dc_imp):
        raise ValueError('Loaded double counting has a different number of '
                         + 'correlated shells than current calculation.')

    has_double_counting_changed = False
    for loaded_dc_shell, calc_dc_shell in zip(loaded_dc_imp, sum_k.dc_imp):
        if sorted(loaded_dc_shell.keys()) != sorted(calc_dc_shell.keys()):
            raise ValueError('Loaded double counting has a different block '
                             + 'structure than current calculation.')

        for channel in loaded_dc_shell.keys():
            if not np.allclose(loaded_dc_shell[channel], calc_dc_shell[channel],
                               atol=1e-4, rtol=0):
                has_double_counting_changed = True
                break

    # Sets initial Sigma
    start_sigma = loaded_sigma

    if not has_double_counting_changed:
        print('DC remained the same. Using loaded Sigma as initial Sigma.')
        return start_sigma

    # Uses the SumkDFT add_dc routine to correctly substract the DC shift
    sum_k.put_Sigma(start_sigma)
    calculated_dc_imp = sum_k.dc_imp
    sum_k.dc_imp = [{channel: np.array(loaded_dc_shell[channel]) - np.array(calc_dc_shell[channel])
                     for channel in loaded_dc_shell}
                    for calc_dc_shell, loaded_dc_shell in zip(sum_k.dc_imp, loaded_dc_imp)]
    start_sigma = sum_k.add_dc()
    start_sigma = _sumk_sigma_to_solver_struct(sum_k, start_sigma)

    # Prints information on correction of Hartree shift
    first_block = sorted(key for key, _ in loaded_sigma[0])[0]
    print('DC changed, initial Sigma is the loaded Sigma with corrected Hartree shift:')
    print('    Sigma for imp0, block "{}", orbital 0 '.format(first_block)
          + 'shifted from {:.3f} eV '.format(loaded_sigma[0][first_block].data[0, 0, 0].real)
          + 'to {:.3f} eV'.format(start_sigma[0][first_block].data[0, 0, 0].real))

    # Cleans up
    sum_k.dc_imp = calculated_dc_imp
    [sigma_freq.zero() for sigma_freq in sum_k.Sigma_imp]

    return start_sigma


def determine_dc_and_initial_sigma(general_params, gw_params, advanced_params, sum_k,
                                   archive, iteration_offset, G_loc_all, solvers,
                                   solver_type_per_imp):
    """
    Determines the double counting (DC) and the initial Sigma. This can happen
    in five different ways:
    * Calculation resumed: use the previous DC and the Sigma of the last complete calculation.

    * Calculation initialized with load_sigma: use the DC and Sigma from the previous file.
      If the DC changed (and therefore the Hartree shift), the initial Sigma is adjusted by that.

    * New calculation, with DC: calculate the DC, then initialize the Sigma as the DC,
      effectively starting the calculation from the DFT Green's function.
      Also breaks magnetic symmetry if calculation is magnetic.

    * New calculation, without DC: Sigma is initialized as 0,
      starting the calculation from the DFT Green's function.

    Parameters
    ----------
    general_params : dict
        general parameters as a dict
    gw_params : dict
        GW parameters as a dict
    advanced_params : dict
        advanced parameters as a dict
    sum_k : SumkDFT object
        Sumk object with the information about the correct block structure
    archive : HDFArchive
        the archive of the current calculation
    iteration_offset : int
        the iterations done before this calculation
    G_loc_all : Gf
        local Green function for all shells
    solvers : list
        list of Solver instances

    Returns
    -------
    sum_k : SumkDFT object
        the SumkDFT object, updated by the initial Sigma and the DC
    solvers : list
        list of Solver instances, updated by the initial Sigma

    """
    start_sigma = None
    last_g0 = None
    density_mat_dft = [G_loc_all[iineq].density() for iineq in range(sum_k.n_inequiv_shells)]
    if mpi.is_master_node():
        # Resumes previous calculation
        if iteration_offset > 0:
            print('\nFrom previous calculation:', end=' ')
            start_sigma, sum_k.dc_imp, sum_k.dc_energ, last_g0,  _ = _load_sigma_from_h5(archive, -1)
            if general_params['csc'] and not general_params['dc_dmft']:
                sum_k = calculate_double_counting(sum_k, density_mat_dft, general_params, gw_params,
                                                  advanced_params, solver_type_per_imp, G_loc_all)
        # Loads Sigma from different calculation
        elif general_params['load_sigma']:
            print('\nFrom {}:'.format(general_params['path_to_sigma']), end=' ')
            with HDFArchive(general_params['path_to_sigma'], 'r') as sigma_archive:
                (loaded_sigma, loaded_dc_imp, _,
                 _, loaded_density_matrix) = _load_sigma_from_h5(sigma_archive, general_params['load_sigma_iter'])

            # Recalculate double counting in case U, J or DC formula changed
            if general_params['dc']:
                if general_params['dc_dmft']:
                    sum_k = calculate_double_counting(sum_k, loaded_density_matrix, general_params, gw_params,
                                                      advanced_params, solver_type_per_imp, G_loc_all)
                else:
                    sum_k = calculate_double_counting(sum_k, density_mat_dft, general_params, gw_params,
                                                      advanced_params, solver_type_per_imp, G_loc_all)

            start_sigma = _set_loaded_sigma(sum_k, loaded_sigma, loaded_dc_imp, general_params)

        # Sets DC as Sigma because no initial Sigma given
        elif general_params['dc']:
            sum_k = calculate_double_counting(sum_k, density_mat_dft, general_params, gw_params,
                                              advanced_params, solver_type_per_imp, G_loc_all)

            # initialize Sigma from sum_k
            start_sigma = [sum_k.block_structure.create_gf(ish=iineq, gf_function=Gf, space='solver',
                                                           mesh=sum_k.mesh)
                           for iineq in range(sum_k.n_inequiv_shells)]
            for icrsh in range(sum_k.n_inequiv_shells):
                dc_pot = sum_k.block_structure.convert_matrix(sum_k.dc_imp[sum_k.inequiv_to_corr[icrsh]],
                                                               ish_from=sum_k.inequiv_to_corr[icrsh],
                                                               space_from='sumk', space_to='solver')

                if (general_params['magnetic'] and general_params['magmom'] and sum_k.SO == 0):
                    # if we are doing a magnetic calculation and initial magnetic moments
                    # are set, manipulate the initial sigma accordingly
                    fac = general_params['magmom'][icrsh]

                    # init self energy according to factors in magmoms
                    # if magmom positive the up channel will be favored
                    for spin_channel in sum_k.gf_struct_solver[icrsh].keys():
                        if 'up' in spin_channel:
                            start_sigma[icrsh][spin_channel] << -fac + dc_pot[spin_channel]
                        else:
                            start_sigma[icrsh][spin_channel] << fac + dc_pot[spin_channel]
                else:
                    for spin_channel in sum_k.gf_struct_solver[icrsh].keys():
                        start_sigma[icrsh][spin_channel] << dc_pot[spin_channel]

        # Sets Sigma to zero because neither initial Sigma nor DC given
        elif (not general_params['dc'] and general_params['magnetic']):
            start_sigma = [sum_k.block_structure.create_gf(ish=iineq, gf_function=Gf, space='solver', mesh=sum_k.mesh)
                                for iineq in range(sum_k.n_inequiv_shells)]
            for icrsh in range(sum_k.n_inequiv_shells):
                if (general_params['magnetic'] and general_params['magmom'] and sum_k.SO == 0):
                    mpi.report(f'\n*** Adding magnetic bias to initial sigma for impurity {icrsh} ***')
                    # if we are doing a magnetic calculation and initial magnetic moments
                    # are set, manipulate the initial sigma accordingly
                    fac = general_params['magmom'][icrsh]

                    # if magmom positive the up channel will be favored
                    for spin_channel in sum_k.gf_struct_solver[icrsh].keys():
                        if 'up' in spin_channel:
                            start_sigma[icrsh][spin_channel] << -fac
                        else:
                            start_sigma[icrsh][spin_channel] << fac
        else:
            start_sigma = [sum_k.block_structure.create_gf(ish=iineq, gf_function=Gf, space='solver', mesh=sum_k.mesh)
                           for iineq in range(sum_k.n_inequiv_shells)]

    # Adds random, frequency-independent noise in zeroth iteration to break symmetries
    if not np.isclose(general_params['noise_level_initial_sigma'], 0) and iteration_offset == 0:
        if mpi.is_master_node():
            for start_sigma_per_imp in start_sigma:
                for _, block in start_sigma_per_imp:
                    noise = np.random.normal(scale=general_params['noise_level_initial_sigma'],
                                             size=block.data.shape[1:])
                    # Makes the noise hermitian
                    noise = np.broadcast_to(.5 * (noise + noise.T), block.data.shape)
                    block += Gf(indices=block.indices, mesh=block.mesh, data=noise)

    # bcast everything to other nodes
    sum_k.dc_imp = mpi.bcast(sum_k.dc_imp)
    sum_k.dc_energ = mpi.bcast(sum_k.dc_energ)
    start_sigma = mpi.bcast(start_sigma)
    last_g0 = mpi.bcast(last_g0)
    # Loads everything now to the solver
    for icrsh in range(sum_k.n_inequiv_shells):
        solvers[icrsh].Sigma_freq = start_sigma[icrsh]
        if last_g0:
            solvers[icrsh].G0_freq = last_g0[icrsh]

    # Updates the sum_k object with the Matsubara self-energy
    sum_k.put_Sigma([solvers[icrsh].Sigma_freq for icrsh in range(sum_k.n_inequiv_shells)])

    # load sigma as first guess in the hartree solver if applicable
    for icrsh in range(sum_k.n_inequiv_shells):
        # TODO:
        # should this be moved to before the solve() call? Having it only here means there is a mismatch
        # between the mixing at the level of the solver and the sumk (solver mixes always 100%)
        if solver_type_per_imp[icrsh] == 'hartree':
            mpi.report(f"SOLID_DMFT: setting first guess hartree solver for impurity {icrsh}")
            solvers[icrsh].triqs_solver.reinitialize_sigma(start_sigma[icrsh])

    return sum_k, solvers
