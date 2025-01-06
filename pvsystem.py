import pvlib
import logging
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def make_pv_system(latitude: float, longitude: float) -> pvlib.modelchain.ModelChain:
    location = pvlib.location.Location(latitude=latitude, longitude=longitude)
    mount = pvlib.pvsystem.FixedMount(surface_tilt=latitude)

    # TODO: pass module and inverter as args to the function
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    module = sandia_modules['AstroPower_AP_1206___1998_'] #  From B. Bouzidi paper (2013)
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_'] # https://www.solaris-shop.com/abb-micro-0-25-i-outd-us-208-240-250w-microinverter/
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    array = pvlib.pvsystem.Array(
        # TODO: provide albedo
        mount=mount,
        module_parameters=module,
        temperature_model_parameters=temperature_model_parameters,
    )
    system = pvlib.pvsystem.PVSystem(arrays=[array], inverter_parameters=inverter)
    return pvlib.modelchain.ModelChain(system, location)