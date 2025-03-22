import pvlib
import logging
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

COST_PER_PANEL = 330 # From Bouzidi paper  
COST_POWER_INVERTER = 411
COST_PUMP = 750
COST_PER_M3_TANK = 285 # From Ahmed and Demirci paper

def make_pv_system(latitude: float, longitude: float) -> pvlib.modelchain.ModelChain:
    location = pvlib.location.Location(latitude=latitude, longitude=longitude)
    mount = pvlib.pvsystem.FixedMount(surface_tilt=latitude)

    # TODO: pass module and inverter as args to the function
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    #  From B. Bouzidi paper (2013)
    # 500$/unit according to https://library.uniteddiversity.coop/Energy/Home.Power.Magazine/Home_Power_Magazine_078.PDF
    # 330$/unit according to Bouzidi paper
    module = sandia_modules['AstroPower_AP_1206___1998_'] 



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


def appraise_system(number_solar_panels: int, tank_capacity: float) -> float:
    """
    Calculate a cost metric for the system configuration.
    """
    cost = COST_POWER_INVERTER + COST_PUMP
    cost += tank_capacity * COST_PER_M3_TANK
    cost += number_solar_panels * COST_PER_PANEL
    return cost / 400