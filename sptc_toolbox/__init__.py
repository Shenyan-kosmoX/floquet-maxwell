"""Legume adapters and explicitly scoped open Floquet extensions.

The installed Legume package is used without modifications. Time-harmonic
assembly is reused from the independently tested ptc_fig4 core.
"""

# Static-solver libraries are loaded by their adapters, not by this public
# entry point. Importing the package does not run a geometry or simulation.
from .mode_package import ModePackage
from .modal_adapter import ModalPencilModel, modal_scattering_system, frozen_model_ports
from .qnm import generalized_modes, simple_pole_residue, PoleExpansion
from .dispersive import DispersiveMaxwell, LorentzOscillator, solve_dispersive_floquet
from .scattering import PeriodicStateSpace, ChannelFluxMetric, solve_floquet_scattering
from .transient import TransientResponse, solve_transient_state_space, integrated_flux
from .modulation import RegionModulation, bloch_momentum_selection
from .spatial_drive import SpatialFourierTerm, SpatialFourierDrive, CellPeriodicDrive
from .patterned_drive import NativeCellwiseProjection, PatternedElectricMassModel, project_cellwise_slab
from .tensor_projection import NativeTensorProjection, project_tensor_slab
from .slab_projection import SlabRegion, SlabProjection, project_slab
from .slab_model import SlabMaterialModel
from .slab_study import solve_slab_bloch, solve_slab_local
from .slab_fields import reconstruct_slab_fields, reconstruct_from_spectrum
from .radiation_assessment import native_signed_sampler, assess_frozen_radiation
from .slab_memory import build_slab_memory
from .dynamic_window import DynamicTolerance, study_slab_dynamic_window
from .traveling_slab import TravelingSpec, project_traveling_slab, export_sector_radiation
from .field_chart import native_coefficients_to_physical, register_native_field_record
from .slab_dispersion import RegionLorentzMaterial, SlabDispersiveModel
from .traveling_open import primitive_momentum_orbit, TravelingMarkovBath, evolve_traveling_markov
from .physical_ports import StaticPortReference
from .dispersive_open import DispersiveOpenModel, build_dispersive_open
from .traveling_memory import build_traveling_memory, evolve_traveling_memory
from .slab_ports import NativeSlabPortModel
from .reciprocal_scattering import ScatteringFields, first_order_conversion
from .effective_dynamics import evolve_electric_mass, propagate_floquet_subspace
from .wavepacket_regularities import TravelingMassFrame, bloch_cell_transform
from .scattering_stability import assess_periodic_stability
from .real_frequency_channels import combine_real_frequency_channels
from .slab_excitation import CausalSlabBackground, NativeSlabPulseResponse, evolve_native_slab_pulse
from .green_scattering import StaticGreenResponse, StaticGreenSystem, ModulationKernel, solve_green_floquet
from .native_green import NativeGreenModel
from .projected_green import (FieldBasis, ProjectedGreenSample, ProjectedGreenDataset,
                              ProjectedModulation, ProjectedOperatorModulation, ProjectedModulationSum, weighted_svd_field_basis,
                              enrich_field_basis, solve_projected_green,
                              YEE_CURRENT_CONVENTION, yee_frequency)
from .yee_material import YeeMaterialDerivative, NativeYeeSymmetryChart

__all__ = ['ModePackage','ModalPencilModel','modal_scattering_system','frozen_model_ports',
    'generalized_modes','simple_pole_residue','PoleExpansion','DispersiveMaxwell',
    'LorentzOscillator','solve_dispersive_floquet','PeriodicStateSpace','ChannelFluxMetric',
    'solve_floquet_scattering','TransientResponse','solve_transient_state_space','integrated_flux',
    'RegionModulation','bloch_momentum_selection',
    'SpatialFourierTerm','SpatialFourierDrive','CellPeriodicDrive',
    'NativeCellwiseProjection','PatternedElectricMassModel','project_cellwise_slab',
    'NativeTensorProjection','project_tensor_slab',
    'SlabRegion','SlabProjection','project_slab','SlabMaterialModel','solve_slab_bloch','solve_slab_local',
    'reconstruct_slab_fields','reconstruct_from_spectrum',
    'native_signed_sampler','assess_frozen_radiation','build_slab_memory',
    'DynamicTolerance','study_slab_dynamic_window',
    'TravelingSpec','project_traveling_slab','export_sector_radiation',
    'native_coefficients_to_physical','register_native_field_record',
    'RegionLorentzMaterial','SlabDispersiveModel',
    'primitive_momentum_orbit','TravelingMarkovBath','evolve_traveling_markov','StaticPortReference',
    'DispersiveOpenModel','build_dispersive_open','build_traveling_memory',
    'evolve_traveling_memory','NativeSlabPortModel','ScatteringFields','first_order_conversion',
    'evolve_electric_mass','propagate_floquet_subspace','TravelingMassFrame','bloch_cell_transform',
    'assess_periodic_stability','combine_real_frequency_channels','CausalSlabBackground',
    'NativeSlabPulseResponse','evolve_native_slab_pulse','StaticGreenResponse',
    'StaticGreenSystem','ModulationKernel','solve_green_floquet','NativeGreenModel',
    'FieldBasis','ProjectedGreenSample','ProjectedGreenDataset','ProjectedModulation','ProjectedOperatorModulation','ProjectedModulationSum',
    'weighted_svd_field_basis','enrich_field_basis','solve_projected_green',
    'YEE_CURRENT_CONVENTION','yee_frequency','YeeMaterialDerivative','NativeYeeSymmetryChart']

