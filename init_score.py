#    Copyright (C) 2010 Adam Charette-Castonguay
#    the University of Queensland
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
__author__ = "Adam Charette-Castonguay"

import geopandas as gpd
import pandas as pd
import numpy as np
import time
import multiprocessing
import logging
from affine import Affine
from rasterio import features
import rasterio

pd.options.mode.chained_assignment = None
######################### Load tables #########################

grass_energy = pd.read_csv("tables/grass_energy.csv")  # Load energy in grasses
# beef_production = pd.read_csv("tables/beef_production.csv", index_col="Code")  # Load country-level beef supply
fertiliser_prices = pd.read_csv("tables/fertiliser_prices.csv")  # Load fertiliser prices
nutrient_req_grass = pd.read_csv("tables/nutrient_req_grass.csv")  # Load nutrient requirement for grasses
beef_demand = pd.read_csv("tables/beef_demand.csv")  # Load country-level beef demand
sea_distances = pd.read_csv("tables/sea_distances.csv")  # Load averaged distances between countries
sea_t_costs = pd.read_csv("tables/sea_t_costs.csv")  # Load sea transport costs
energy_conversion = pd.read_csv("tables/energy_conversion.csv")  # Load energy conversion table
fuel_cost = pd.read_csv("tables/fuel_costs.csv")  # fuel cost

crop_area = pd.read_csv("tables/crop_area.csv")  # proportion of crop areas by country
feed_energy = pd.read_csv("tables/feed_energy.csv")  # ME in different feeds
partner_me = pd.read_csv("tables/partner_me.csv")  # Weighted average of ME to meat conversion factor in export partner countries
potential_yields = pd.read_csv("tables/potential_yields.csv")  # Potential yields by climate bins
yield_fraction = pd.read_csv("tables/yield_fraction.csv")  # Fraction yield gap
percent_exported = pd.read_csv("tables/percent_exported.csv")  # Fraction of exported feed
feedprices = pd.read_csv("tables/feedprices.csv")  # Crop prices
crop_emissions_factors = pd.read_csv("tables/emissions_factors.csv")  # N2O emission factors from N for crops
feedpartners = pd.read_csv("tables/feedpartners.csv")  # Trade partners for each feed
expcosts = pd.read_csv("tables/expcosts.csv")  # Export cost of feeds
sea_dist = pd.read_csv("tables/sea_dist.csv")  # Sea distances matrix
exp_access = pd.read_csv("tables/partner_access.csv")  # Access to market in importing country
fuel_partner = pd.read_csv("tables/fuel_partner.csv")  # Fuel cost in partner countries
fertiliser_requirement = pd.read_csv("tables/fertiliser_requirement.csv")  # fertiliser requirement per crop production
energy_efficiency = pd.read_csv("tables/energy_efficiency.csv")  # Energy efficiency
crop_residues= pd.read_csv("tables/crop_residues.csv")  # Residue to product ratio
residue_energy= pd.read_csv("tables/residue_energy.csv")  # Energy in crop residues
stover_frac = pd.read_csv("tables/stover_frac.csv")  # Fraction of stover feed for beef cattle vs all livestock
sc_change = pd.read_csv("tables/sc_change.csv")  # Fraction of stover feed for beef cattle vs all livestock
feed_composition= pd.read_csv("tables/feed_composition.csv")  # Energy efficiency
beefprices = pd.read_csv("tables/beef_price.csv", usecols = ['ADM0_A3', 'price'])
feed_compo = pd.read_csv("tables/grain_stover_compo.csv")
aff_costs = pd.read_csv("tables/aff_costs.csv")
crop_emissions = pd.read_csv("tables/crop_emissions.csv")

foddercrop_area = pd.read_csv("tables/foddercrop_area.csv")

######################### Set parameters #########################

# Creat list of grazing options
grass_cols = []
for i in ["0250", "0375", "0500"]:
    for n in ["000", "050", "200"]:
        grass_cols.append("grass_" + i + "_N" + n)

stover_removal = 0.4  # Availability of crop residues

# Grass N20 emission_factors from N application from Gerber et al 2016
grass_n2o_factor = 0.007

fuel_efficiency = 0.4 # fuel efficiency in l/km
truck_emission_factor = 2.6712 # Emissions factor for heavy trucks (kg CO2/l)
sea_emissions =  0.048  # Emissions factor for heavy trucks (kg CO2/ton-km)
dressing = 0.625 # dressing percentage

# Energy consumption related to processing and packaging, MJ·kg CW-1,  from GLEAM
process_pack = 1.45

# Create list monthly effective temperature column names
months = ["efftemp0" + str(i) for i in range(1, 10)] + ["efftemp10", "efftemp11", "efftemp12"]

# column names for optimal costs/emissions sources
new_colnames = {'production': '_meat',
                'enteric': '_meth',
                'manure': '_manure',
                'export_emissions': '_exp_emiss',
                'export_cost': '_exp_costs',
                'transp_emission': '_trans_emiss',
                'transp_cost': '_trans_cost',
                'total_cost': '_tot_cost',
                'total_emission': '_ghg',
                'n2o_emissions': '_n2o',
                'production_cost': '_cost',
                'agb_change': '_agb_change',
                'opportunity_cost': '_opp_cost',
                'bgb_change': '_bgb_change',
                'processing_energy': '_process_energy',
                'compensation': '_compensation',
                'beef_area': '_area',
                'establish_cost': '_est_cost',
                'biomass': '_BM'
                }

# List of all 16 crops, used to determine available area on cell to grow feed
crop_list = ['barley', 'cassava', 'groundnut', 'maize', 'millet', 'oilpalm', 'potato', 'rapeseed', 'rice', 'rye',
             'sorghum', 'soybean', 'sugarbeet', 'sugarcane', 'sunflower', 'wheat']

# List of all 9 main feed crops
# foddercrop_list = ['barley', 'maize', 'millet', 'rapeseed', 'rice', 'rye', 'sorghum', 'soybean', 'wheat']
foddercrop_list = ['barley', 'maize', 'rapeseed', 'rice', 'sorghum', 'soybean', 'wheat']

def eac(cost, rate = 0.07, lifespan = 30.):
    """
    Function to annualize a cost based on a discount rate and a lifespan

    Arguments:
    cost (float) -> One-off cost to annualise
    rate (float)-> Discount rate, default = 7% (Wang et al 2016 Ecol Econ.)
    lifespan (float)-> Time horizon, default: 30 commonly practiced in agricultural investment (Wang et al 2016 Ecol Econ.)

    Output: returns the annualised cost as a float
    """

    if rate == 0: # For emissions -> no discount rate
        return cost/lifespan
    else:
        return (cost * rate)/(1-(1+rate)**-lifespan)

def weighted_score(feats, l, lam, optimisation_method):

    """
    Function to calculate score based on relative costs and emissions

    Arguments:
    feats (dataframe) -> Dataframe in which to look for land use to optimise
    l (str)-> land use for which to calculate score
    lam (float)-> Lambda weight for optimisation
    optimisation_method (str)-> Method for optimisation, either weighted sum or carbon pricing

    Output: returns the annualised cost as a float
    """
    ntrips = (feats[l + '_meat'] / int(15) + 1) * 2

    feats[l + '_itl_trans_cost'] = ntrips * feats["distance_port"] * \
                               feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                        how='left', left_on='ADM0_A3',
                                                        right_on='ADM0_A3')['Diesel'].values * fuel_efficiency / 1000.

    # Calculate transport costs as a function of quantity traded
    feats[l + '_itl_exp_costs'] = feats[l + '_meat'] * feats[['ADM0_A3']].merge(sea_t_costs[['ADM0_A3', 'tcost']],
                                                                            how='left')['tcost'].values

    # Transport emissions to port
    feats[l + '_itl_trans_emiss'] = ntrips * feats["distance_port"] * fuel_efficiency * truck_emission_factor / 1000.

    # Transport emissions by sea
    feats[l + '_itl_exp_emiss'] = feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')[
                                  'ave_distance'].values * feats[l + '_meat'] * sea_emissions / 1000.

    feats[l + '_exp_emiss'] = -feats[l + '_trans_emiss'] + feats[l + '_itl_trans_emiss'] + feats[l + '_itl_exp_emiss']
    feats[l + '_exp_costs'] = -feats[l + '_trans_cost'] + feats[l + '_itl_trans_cost'] + feats[l + '_itl_exp_costs']


    # Annualise establishment cost, same for pasture or crops, adjusted by the ratio of area used:cell area
    feats[l + '_est_cost'] = eac(feats['est_cost'] * (feats[l + '_area'] - (feats['current_grazing'].fillna(0).values + feats['current_cropping'].fillna(0).values)) / feats['suitable_area'])

    # Calculate opportunity cot in '000 USD: ag value (USD/ha) x area used (ha)
    feats[l + '_opp_cost'] = feats['opp_cost'].astype(float) / 1000. * feats[l + '_area']

    # Calculate current production: cell area (ha) x meat (kg/km2) x kg/km2-t/ha conversion
    current_production = feats['area'].values * feats['bvmeat'].values * 1e-5

    # Get price of beef in '000 USD
    beef_price = feats[['ADM0_A3']].merge(beefprices, how='left')['price'].values / 1000.

    # Compensation for loss of revenues from beef ('000 USD) = max(0, beef prices ('000 USD) x (current production - new production))
    feats[l + '_compensation'] = np.maximum(0, beef_price * (current_production - feats[l + '_meat'].values))
    del beef_price, current_production

    # Emissions for processing and packaging energy = meat (t) * process energy (MJ/kg) * energy efficiency (kg CO2/kg)
    feats[l + '_process_energy'] = feats[l + '_meat'].values * process_pack * \
                                   feats[['ADM0_A3']].merge(energy_efficiency, how='left')['energy'].fillna(0).values

    # Total costs ('000 USD) = Establishment cost (annualised '000 USD) + production cost + transport cost + opportunity cost + cost of afforestation
    feats[l + '_tot_cost'] = feats[l + '_est_cost'] + feats[l + '_cost'] + feats[l + '_trans_cost'] + feats[
        l + '_opp_cost'] - feats['aff_cost']
    # + feats[l + '_compensation']

    # Annual emissions (t CO2 eq) = Fertiliser N2O + Enteric CH4 + Manure N2O + transport CO2 + Processing CO2
    flow = feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_manure'] + feats[l + '_trans_emiss'] + feats[
        l + '_process_energy']

    # Carbon stock = Current C stock (t/ha) * area (ha) * C-CO2 conversion - remaining C stock (t)
    # agb_change = feats['agb_spawn'] * feats['suitable_area'] * 3.67 - feats[l + '_cstock']

    # Annualise the loss of carbon stock
    feats[l + '_agb_change'] = eac(feats[l + '_cstock'], rate=0)

    # Total GHG emissions = Above ground carbon change + below ground carbon change + annual emissions - afforestation potential - soil carbon potential
    feats[l + '_ghg'] = flow + feats[l + '_agb_change'] + feats[l + '_bgb_change'] - feats['opp_aff'] - feats[
        'opp_soc']

    # Set export cost and emissions to 0
    feats[l + '_exp_emiss'] = 0
    feats[l + '_exp_costs'] = 0

    # Calculate relative GHG (GHG/meat)(t CO2 eq)/Meat (ton)
    rel_ghg = np.where(feats[l + '_meat'] < 1, np.NaN, feats[l + '_ghg'] / (feats[l + '_meat']))

    # Calculate relative Cost (Cost/meat) Cost ('000 USD)/Meat (ton)
    rel_cost = np.where(feats[l + '_meat'] < 1, np.NaN,
                        feats[l + '_tot_cost'] / (feats[l + '_meat']))

    if optimisation_method == 'carbon_price':
        lam = lam / 1000.

    feats[l + '_score'] = (rel_ghg * (1 - lam)) + (rel_cost * lam)

    return feats

def scoring(feats, optimisation_method, crop_yield, lam, beef_yield, aff_scenario, logger, feed_option, landuses):
    """
    Finds the best landuse for each cell in the partition and returns the partition

    Arguments:
    feats (pandas dataframe) -> Main dataframe
    optimisation_method (str)-> Method for optimisation ('weighted_sum' or 'carbon_price')
    crop_yield (int)-> Scenario of crop yield (0 = current, 1 = no yield gap)
    lam (float)-> Lambda weight ([0,1])
    beef_yield (str)-> Scenario of beef yield ('me_to_meat' = current, 'max_yield' = no yield gap)
    logger (RootLogger) -> logger defined in main function
    feed_option (str)-> folder where the output file is exported

    Output: returns a gridded dataframe with scores
    """

    # Adjust yield fraction based on yield gap reduction scenario
    yield_fraction[crop_list] = yield_fraction[crop_list] + crop_yield

    # Cap yield fraction to 1 (cannot be high than attainable yield)
    yield_fraction[crop_list] = yield_fraction[crop_list].where(~(yield_fraction[crop_list] > 1), other=1)

    ################## Calculate available area ##################
    feats['available_area'] = np.where(feats['newarea'].values == 1,
                                       feats['suitable_area'].values - feats['net_fodder_area'].values,
                                       feats['current_grazing'].fillna(0).values + feats['current_cropping'].fillna(
                                           0).values)

    # Limit cells where there is area available
    feats = feats.loc[feats['available_area'] > 0]

    ################## SOC and AGB change from removing current beef ##################
    # Get percentage change of grassland and cropland to 'original' ecoregion
    soc_change = feats[['ecoregions']].merge(sc_change[['code', 'grassland', 'cropland']], how='left', 
                                             left_on='ecoregions', right_on='code')

    # 'Opportunity cost' of soil carbon = sum over land uses of current area of land use * current soil carbon * percentage change * negative emission (-1)
    opp_soc = (feats['current_grazing'].fillna(0).values * feats['bgb_spawn'].values * soc_change[
        'grassland'].values + feats['current_cropping'].fillna(0).values * feats['bgb_spawn'].values * soc_change['cropland'].values) * -1

    # Make sure that opportunity cost of soil carbon is only where there currently is beef production and convert C to CO2
    opp_soc = np.where(feats.newarea.values == 0, opp_soc, 0) * 3.67

    #Select forest types (boreal, temperate, tropical) based on ecoregions
    forest_type = np.select(
        [np.isin(feats.ecoregions, [1, 2, 3, 7, 9]), np.isin(feats.ecoregions, [4, 5, 8, 12]),
         np.isin(feats.ecoregions, [6, 10, 11])],
        [1, 2, 3], default=0)

    # Calculate regrowth area (only on cell where beef is currently produced)
    regrowth_area = np.where(feats.newarea.values == 0,
                             feats['current_grazing'].fillna(0).values + feats['current_cropping'].fillna(0).values,
                             0)

    # Calculate max carbon stock on cell based on area of regrowth and max carbon density
    max_cstock = regrowth_area * feats['potential_carbon'].values * -1

    if aff_scenario == 'noaff':
        # Afforestation is 0
        opp_aff = np.zeros_like(feats.ADM0_A3, dtype='int8')
        # Afforestation cost is 0
        feats['aff_cost'] = np.zeros_like(feats.ADM0_A3, dtype='int8')
        # "Opportunity cost" of soil carbon is 0
        opp_soc = 0
    elif aff_scenario == 'nataff':
        # Parameters for carbon growth curves for different forest types and natural regrowth
        k = np.select([forest_type == 1, forest_type == 2, forest_type == 3],[0.02, 0.02, 0.02], default=0)
        p = np.select([forest_type == 1, forest_type == 2, forest_type == 3],[2, 5, 5], default=0)
        t = 30.

        # Carbon stock after t = 30 years
        opp_aff = max_cstock * (1 - np.exp(-k * t)) ** p

        # Afforestation cost is 0
        feats['aff_cost'] = np.zeros_like(feats.ADM0_A3, dtype='int8')

    elif aff_scenario == 'manaff':

        # Parameters for carbon growth curves for different forest types and managed regrowth
        k = np.select([forest_type == 1, forest_type == 2, forest_type == 3],[0.059, 0.2, 0.05], default=0)
        p = np.select([forest_type == 1, forest_type == 2, forest_type == 3],[1.65, 4, 4], default=0)
        t = 30.

        # Carbon stock after t = 30 years
        opp_aff = max_cstock * (1 - np.exp(-k * t)) ** p

        # Initial cost and long term rotation annual cost (in '000 USD)
        initial_aff_cost = feats[['ADM0_A3']].merge(aff_costs, how='left')['initial'].values/1000. * regrowth_area
        annual_aff_cost = feats[['ADM0_A3']].merge(aff_costs, how='left')['annual'].values/1000. + regrowth_area

        # Afforestation costs = Annualise initial cost + annual cost
        feats['aff_cost'] = eac(initial_aff_cost) + annual_aff_cost

    else:
        logger.info("Afforestation scenario {} not in choices".format(aff_scenario))

    # Make sure that opportunity cost of afforestation is greater than 0 & Convert C to CO2
    opp_aff = np.where(feats.newarea.values == 0, opp_aff, 0) * 3.67

    # Annualise afforestation and opp cost of soil carbon
    feats['opp_aff'] = eac(opp_aff, rate=0)
    feats['opp_soc'] = eac(opp_soc, rate=0)

    # ----------------------- #
    k = np.select([forest_type == 1, forest_type == 2, forest_type == 3], [0.02, 0.02, 0.02], default=0)
    p = np.select([forest_type == 1, forest_type == 2, forest_type == 3], [2, 5, 5], default=0)
    t = 30.
    opp_aff = max_cstock * (1 - np.exp(-k * t)) ** p
    opp_aff = np.where(feats.newarea.values == 0, opp_aff, 0) * 3.67
    feats['opp_nataff'] = eac(opp_aff, rate=0)

    k = np.select([forest_type == 1, forest_type == 2, forest_type == 3], [0.059, 0.2, 0.05], default=0)
    p = np.select([forest_type == 1, forest_type == 2, forest_type == 3], [1.65, 4, 4], default=0)
    t = 30.
    opp_aff = max_cstock * (1 - np.exp(-k * t)) ** p
    opp_aff = np.where(feats.newarea.values == 0, opp_aff, 0) * 3.67
    feats['opp_manaff'] = eac(opp_aff, rate=0)

    initial_aff_cost = feats[['ADM0_A3']].merge(aff_costs, how='left')['initial'].values / 1000. * regrowth_area
    annual_aff_cost = feats[['ADM0_A3']].merge(aff_costs, how='left')['annual'].values / 1000. + regrowth_area
    feats['affor_cost'] = eac(initial_aff_cost) + annual_aff_cost

    opp_soc = (feats['current_grazing'].fillna(0).values * feats['bgb_spawn'].values * soc_change[
        'grassland'].values + feats['current_cropping'].fillna(0).values * feats['bgb_spawn'].values * soc_change['cropland'].values) * -1
    opp_soc = np.where(feats.newarea.values == 0, opp_soc, 0) * 3.67
    feats['opport_soc'] = eac(opp_soc, rate=0)

    # ----------------------- #



    del opp_aff, soc_change, opp_soc

    if feed_option in ['v1', 'v2']:

        for l in grass_cols:

            # For grazing, convert all area
            feats[l + '_area'] = feats['available_area'].values

            # Calculate biomass consumed (ton) = (grazed biomass (t/ha) * area (ha))
            # biomass_consumed = feats[l].values * feats['suitable_area'].values
            feats[l + '_BM'] = feats[l].values * feats[l + '_area']

            # Subset energy conversion table to keep grazing systems and ME to meat conversion column.
            # Climate coding: 1 = Temperate, 2 = Arid, 3 = Humid
            subset_table = energy_conversion.loc[energy_conversion.feed == 'grazing'][['group', 'glps', beef_yield,
                                                                                       'curr_methane', 'curr_manure']]

            # Calculate energy consumed ('000 MJ) = biomass consumed (t) * energy in grass (MJ/kg)
            energy = feats[l + '_BM'] * feats.merge(
                grass_energy, how='left', left_on=['region', 'glps'], right_on=['region', 'glps'])['ME'].values

            # Meat production (t) = energy consumed ('000 MJ) * energy conversion (kg/MJ) * dressing (%)
            meat = energy * feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
                                                           right_on=['group', 'glps'])[beef_yield].values * dressing

            # Adjust meat prodution based on effective temperature
            feats[l + '_meat'] = np.sum(np.where(feats[months] < -1, (meat[:, None] - (meat[:, None] * (
                    -0.0182 * feats[months] - 0.0182))) / 12., meat[:, None] / 12.), axis=1)

            # Calculate methane production (ton CO2eq) = biomass consumed (t) * conversion factor (ton CO2eq/ton biomass)
            feats[l + '_meth'] = feats[l + '_BM'] * feats[['group', 'glps']].merge(
                subset_table, how='left', left_on=['group', 'glps'], right_on=['group', 'glps'])['curr_methane'].values

            # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
            feats[l + '_manure'] = feats[l + '_BM'] * feats[['group', 'glps']].merge(
                subset_table, how='left', left_on=['group', 'glps'], right_on=['group', 'glps'])['curr_manure'].values

            # Calculate fertiliser application in tons (0 for rangeland, assuming no N, P, K inputs)
            # Extract N application from column name, convert to ton
            n = int(l.split("_N")[1]) / 1000.

            if n == 0:
                n_applied = 0
                k_applied = 0
                p_applied = 0
            else:
                n_applied = int(l.split("_N")[1]) / 1000. * feats['suitable_area'].values

                k_applied = feats['suitable_area'] * feats[['nutrient_availability']].merge(
                    nutrient_req_grass, how='left')['K'].values * 2.2 / 1000.

                p_applied = feats['suitable_area'] * feats[['nutrient_availability']].merge(
                    nutrient_req_grass, how='left')['P'].values * 1.67 / 1000.

            # Get cost of fertilisers per country (USD/ton)
            fert_costs = feats[['ADM0_A3']].merge(fertiliser_prices, how='left')

            # Get total cost of fertilisers (USD) (N content in nitrate = 80%)
            feats[l + '_cost'] = n_applied * 1.2 * fert_costs['n'].values + k_applied * fert_costs[
                'k'].values + p_applied * fert_costs['p'].values

            # Calculate N20 emissions based on N application = N application (ton) * grass emission factor (%) * CO2 equivalency
            feats[l + '_n2o'] = (n_applied * grass_n2o_factor) * 298

            # Number of trips to market; assuming 15 tons per trip, return
            ntrips = np.ceil(feats[l + '_meat'] / int(15)) * 2

            # Transport cost to market: number of trips * transport cost ('000 US$)
            feats[l + '_trans_cost'] = ntrips * feats[['ADM0_A3']].merge(
                fuel_cost[['ADM0_A3', 'Diesel']], how='left', left_on='ADM0_A3', right_on='ADM0_A3')['Diesel'].values * \
                                       feats["accessibility"] * fuel_efficiency / 1000.

            # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
            feats[l + '_trans_emiss'] = ntrips * feats["accessibility"] * fuel_efficiency * truck_emission_factor /1000.

            # Estimate carbon content as 47.5% of remaining grass biomass. Then convert to CO2 eq (*3.67)
            grazing_intensity = (1 - (int(l.split("_")[1]) / 1000.))
            # feats[l + '_cstock'] = 0.475 * (feats[l + '_BM'] / grazing_intensity * (1 - grazing_intensity)) * 3.67

            # extra_area = np.where(feats[l + '_area'].values <= regrowth_area , 0, feats[l + '_area'].values - regrowth_area)

            feats[l + '_cstock'] = np.where(feats['newarea'] == 1, feats['agb_spawn'] * 3.67 * feats[l + '_area'].values - \
                                   0.475 * ((feats[l].values * feats[l + '_area'].values) / grazing_intensity * (1 - grazing_intensity)) * 3.67,
                                            - 0.475 * ((feats[l].values * (feats[l + '_area'].values - feats[
                                                    'current_grazing'])) / grazing_intensity * ( 1 - grazing_intensity)) * 3.67)


            # Change in soil carbon (t CO2 eq) = change from land use to grassland (%) * current land use area * current soil carbon (t/ha) * C-CO2 conversion * emission (-1)
            bgb_change = ((0.19 * feats['crop_area'] * feats['bgb_spawn']) + (0.08 * feats['tree_area'] * feats[
                'bgb_spawn'])) * 3.67 * -1 * feats[l + '_area'] / feats['suitable_area']

            # Annualise change in soil carbon
            feats[l + '_bgb_change'] = eac(bgb_change, rate=0)

            feats = weighted_score(feats, l, lam, optimisation_method)

        logger.info("Done with grass columns")

        logger.info("Done with grass columns")

        for l in ['grass_grain']:
            # Keep yield fraction for 9 main fodder crops
            fodder_yield_fraction = yield_fraction[['ADM0_A3'] + foddercrop_list]

            # Potential production if all available area is converted to grain
            full_grain_prod = np.nansum(feats['available_area'].values[:, None] * \
                                        feats[['ADM0_A3']].merge(foddercrop_area, how="left").drop('ADM0_A3',
                                                                                                   axis=1).values *
                                        feats[foddercrop_list].values *
                                        feats[['ADM0_A3']].merge(
                fodder_yield_fraction, how="left").drop('ADM0_A3', axis=1).values, axis=1)

            # Start area division with 80% grain, 20% grass
            cropping_prop_area = np.full_like(feats['available_area'].values, 0.8)
            grazing_prop_area = 1 - cropping_prop_area

            # Set grain production = grain prop area * 100% grain production
            grain_production = cropping_prop_area * full_grain_prod

            # Select best grassland landuse for roughage
            grasslu = np.nanargmin(np.ma.masked_array(feats[[g + '_score' for g in grass_cols]].values,
                                                      np.isnan(feats[[g + '_score' for g in
                                                                      grass_cols]].values)), axis=1)

            # Select biomass production from best grassland landuse for roughage
            grasscol = np.take_along_axis(feats[[lu for lu in grass_cols]].values,
                                                        grasslu[:, None], axis=1).flatten()
            
            # Set grass production = grazing proportion area * available area * grazing (t/ha)
            grazing_production = grazing_prop_area * feats['available_area'].values * grasscol

            # Total_production = grain + grazing
            total_production = grain_production + grazing_production

            # Proportion of grain vs total production
            grain_qty_prop = grain_production / total_production
            logger.info(grain_qty_prop[grain_qty_prop > 0.8].shape[0])
            logger.info('Start grain area adjustment')
            pd.set_option('display.max_columns', 8)

            # Keep track of number of rows not meeting constraint
            shape = grain_qty_prop[grain_qty_prop > 0.8].shape[0]

            # As long as there are rows not meeting constraint...
            while shape > 0:
                # df = pd.DataFrame({'Crop_area_frac': cropping_prop_area,
                #                    'Grass_area_frac': grazing_prop_area,
                #                    'grain_qty': grain_production,
                #                    'grass_qty': grazing_production,
                #                    'total': grain_production + grazing_production,
                #                    'grain_bm_prop': grain_qty_prop,
                #                    'grass_bm_prop': 1 - grain_qty_prop})
                # logger.info(df)
                # print(df)
                # If grain BM > 80% of total BM, reduce grain area by 1%, otherwise keep current grain frac
                cropping_prop_area = np.where(grain_qty_prop > 0.8, cropping_prop_area - 0.01, cropping_prop_area)

                # Make sure grain area proportion doesn't get into negatives
                cropping_prop_area = np.where(cropping_prop_area < 0, 0, cropping_prop_area)

                # Update grazing area fraction = 1-cropping area fraction
                grazing_prop_area = 1 - cropping_prop_area
                # Update grain production based on new grain area
                grain_production = cropping_prop_area * full_grain_prod
                # Update grazing biomass based on new grain area
                grazing_production = grazing_prop_area * feats['available_area'].values * grasscol
                # Check the 80% grain in diet constraint
                grain_qty_prop = grain_production / (grain_production + grazing_production)
                # Update number of cells not meeting constraint
                shape = grain_qty_prop[grain_qty_prop > 0.8].shape[0]

                # logger.info('Number of cells with grain BM fraction exceeding 80% to toal BM: {}'.format(shape))
                # df = pd.DataFrame({'Crop_area_frac': cropping_prop_area,
                #                    'Grass_area_frac': grazing_prop_area,
                #                    'grain_qty': grain_production,
                #                    'grass_qty': grazing_production,
                #                    'total': grain_production + grazing_production,
                #                    'grain_bm_prop': grain_qty_prop,
                #                    'grass_bm_prop': 1 - grain_qty_prop})
                # logger.info(df)

            logger.info('End grain area adjustment')

            # Set final grass, grain and total areas
            grain_area = feats['available_area'].values * cropping_prop_area
            grass_area = feats['available_area'].values * grazing_prop_area
            feats[l + '_area'] = grain_area + grass_area

            # Set biomass consumed
            feats['grain_grassBM'] = grass_area * grasscol

            grain_prod = grain_area[:, None] * feats[['ADM0_A3']].merge(foddercrop_area, how="left").drop(
                'ADM0_A3', axis=1).values * feats[foddercrop_list].values *\
                         feats[['ADM0_A3']].merge(fodder_yield_fraction, how="left").drop(
                'ADM0_A3', axis=1).values

                #          feats[['climate_bin']].merge(fodder_potential_yields, how="left").drop(
                # 'climate_bin', axis=1).values * \

            feats['grain_grainBM'] = np.nansum(grain_prod, axis=1)
            feats[l + '_BM'] = feats['grain_grainBM'] + feats['grain_grassBM']

            # Biomass consumed for domestic production (t) = actual production (t) x (1 - fraction exported feed)
            biomass_dom = grain_prod * (1 - feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3'] + foddercrop_list], 
                                                                     how="left").drop('ADM0_A3', axis=1).values)

            # Biomass consumed for domestic production (t) = actual production (t) x fraction exported feed
            biomass_exported = grain_prod * feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3'] + foddercrop_list],
                                                                     how="left").drop('ADM0_A3', axis=1).values

            # Subset ME in conversion per region and climate
            subset_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['group', 'glps', beef_yield,
                                                                                     'curr_methane', 'curr_manure']]
            
            # Meat production (t) = sum across feeds (Domestic biomass (t) x ME in feed (MJ/kd DM)) x ME to beef conversion ratio * dressing (%)      
            grass_meat = feats['grain_grassBM'].values * feats.merge(grass_energy, how='left', left_on=['region', 'glps'], right_on=['region', 'glps'])['ME'].values * \
                   feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'], right_on=['group', 'glps'])[beef_yield].values * dressing

            local_grain_meat = (np.nansum(biomass_dom * feed_energy[foddercrop_list].iloc[0].values[None, :], axis=1) ) * \
                   feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'], right_on=['group', 'glps'])[beef_yield].values * dressing

            # Update meat production after climate penalty
            grass_meat = np.sum(np.where(feats[months] < -1,
                                         (grass_meat[:, None] - (grass_meat[:, None] * (-0.0182 * feats[months] - 0.0182))) / 12.,
                                         grass_meat[:, None] / 12.), axis=1)

            # Differentiate between local meat production and meat production from exported feed
            local_grain_meat = np.sum(np.where(feats[months] < -1,
                                         (local_grain_meat[:, None] - (local_grain_meat[:, None] * (-0.0182 * feats[months] - 0.0182))) / 12.,
                                         local_grain_meat[:, None] / 12.), axis=1)
            local_meat = local_grain_meat + grass_meat

            # Calculate methane produced from local beef production (ton) = biomass consumed (ton) x biomass-methane conversion (ton/ton)
            local_methane = (np.nansum(biomass_dom, axis=1) + feats['grain_grassBM'].values) * \
                            feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
                                                           right_on=['group', 'glps'])['curr_methane'].values

            # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
            local_manure = (np.nansum(biomass_dom, axis=1) + feats['grain_grassBM'].values) * \
                        feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
                                                       right_on=['group', 'glps'])['curr_manure'].values
            
            # Calculate nitrous N2O (ton) = Actual production (ton) x fertiliser requirement (kg) x crop_emission factors (% per thousand)
            # feats[l + '_n2o'] = np.nansum(total_prod * grain_perc* fertiliser_requirement['fertiliser'].values[None, :] * (
            #         crop_emissions_factors['factor'].values[None, :] / 100), axis=1)

            # Get grass area from best grass land use
            grassarea2 = np.take_along_axis(feats[[lu + '_area' for lu in grass_cols]].values, grasslu[:, None], axis=1).flatten()
            # Get grass N2O emissions from best grass land use
            grass_n2o = np.take_along_axis(feats[[lu + '_n2o' for lu in grass_cols]].values, grasslu[:, None], axis=1).flatten()

            # Calculate N2O emissions from fertiliser for fodder crops
            grain_n2o = np.nansum(
                grain_area[:, None] * feats[['ADM0_A3']].merge(foddercrop_area, how="left").drop(
                    'ADM0_A3', axis=1).values * feats[['ADM0_A3']].merge(crop_emissions[['ADM0_A3'] + foddercrop_list],
                                                                         how="left").drop('ADM0_A3', axis=1).values,
                axis=1)

            # Calculate total N2O
            feats[l + '_n2o'] = grain_n2o + (grass_n2o * (grass_area/grassarea2))
            logger.info("Done with local meat production")

            ##### Exported feed #####
            # Create empty arrays to fill in
            meat_abroad = np.zeros_like(feats.ADM0_A3, dtype='float32')
            methane_abroad = np.zeros_like(feats.ADM0_A3, dtype='float32')
            manure_abroad = np.zeros_like(feats.ADM0_A3, dtype='float32')
            exp_costs = np.zeros_like(feats.ADM0_A3, dtype='float32')
            sea_emissions_ls = np.zeros_like(feats.ADM0_A3, dtype='float32')
            emissions_partner_ls = np.zeros_like(feats.ADM0_A3, dtype='float32')
            trancost_partner_ls = np.zeros_like(feats.ADM0_A3, dtype='float32')

            for f in foddercrop_list:  # Loop though feeds
                ### Meat produced abroad
                # Quantity of feed f exported
                if feed_option == "v1":
                    # Qty exported (t) = Suitable area (ha) * crop area fraction * crop yield (t/ha) * yield gap (%) * export fraction
                    qty_exported = ((feats['suitable_area'].values * feats[['ADM0_A3']].merge(
                        foddercrop_area[['ADM0_A3', f + '_area']], how="left")[f + '_area'].values * feats[f].values * feats[['ADM0_A3']].merge(yield_fraction, how="left")[f].values)) * \
                                   feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3', f]], how="left")[f].values
                                     # feats[['climate_bin']].merge(potential_yields[['climate_bin', f]],
                                     #                              how="left")[f].values * \


                if feed_option == "v2":
                    # Qty exported (t) = (Suitable area (ha) * crop area fraction * crop yield (t/ha) * yield gap (%)) - production for other uses (t) * export fraction
                    qty_exported = ((grain_area * feats[['ADM0_A3']].merge(foddercrop_area[['ADM0_A3', f + '_area']], how="left")[
                                         f + '_area'].values * feats[f].values * feats[['ADM0_A3']].merge(yield_fraction, how="left")[f].values)
                                    # - feats['diff_' + f].values
                                    ) * feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3', f]],
                                                            how="left")[f].values

                                     # feats[['climate_bin']].merge(potential_yields[['climate_bin', f ]],
                                     #                              how="left")[f].values * \


                # Make sure there is no negative quantity exported
                qty_exported = np.where(qty_exported < 0, 0, qty_exported)

                # trade partners
                trade_partners = feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.crop == f], how='left').drop(
                    ['ADM0_A3', 'crop'], axis=1).values

                # Meat produced from exported feed (t) = Exported feed (t) * partner fraction (%) * energy in feed ('000 MJ/t) * energy conversion in partner country (t/'000 MJ) * dressing (%)
                meat_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * feed_energy[f].iloc[0] * partner_me['meat'].values[None,
                                                                                      :],
                    axis=1) * dressing

                ### Methane emitted abroad (t CO2 eq) = Exported feed (t) * partner fraction (%) * methane emissions per biomass consumed (t/t)
                methane_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * partner_me["methane"].values[None, :], axis=1)

                ### N2O from manure emitted abroad (t CO2 eq) = Exported feed (t) * partner fraction (%) * N2O emissions per biomass consumed (t/t)
                manure_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * partner_me["manure"].values[None, :], axis=1)

                ### Export cost ('000 USD) = Exported feed (t) * partner fraction (%) * value of exporting crop c to partner p ('000 USD/t)
                exp_costs += np.nansum(qty_exported[:, None] * trade_partners * feats[['ADM0_A3']].merge(
                    expcosts.loc[expcosts.crop == f], how='left').drop(['ADM0_A3', 'crop'], axis=1).values, axis=1)

                ### Sea emissions (t CO2 eq) = Exported feed (t) * partner fraction (%) * sea distance from partner p (km) * sea emissions (kg CO2 eq/t-km) * kg-t conversion
                sea_emissions_ls += np.nansum(qty_exported[:, None] * trade_partners * feats[['ADM0_A3']].merge(
                    sea_dist, how='left').drop(['ADM0_A3'], axis=1).values * sea_emissions, axis=1) / 1000.

                ### Number of local transport cost in importing country
                ntrips_local_transp = qty_exported[:, None] * trade_partners / int(15) * 2

                ### Transport cost in partner country ('000 USD) = trips * accessibility to market in partner country (km) * fuel cost in partner country * fuel efficiency * USD-'000 USD conversion
                trancost_partner_ls += np.nansum(
                    ntrips_local_transp * exp_access['access'].values[None, :] * fuel_partner[
                                                                                     'Diesel'].values[None,
                                                                                 :] * fuel_efficiency / 1000., axis=1)

                ### Transport emissions in partner country (t CO2 eq) = trips * accessibility to market in partner country (km) *
                # fuel efficiency (l/km) * truck emission factor (kg CO2 eq/l) * kg-ton conversion
                emissions_partner_ls += np.nansum(
                    ntrips_local_transp * exp_access['access'].values[None,
                                          :] * fuel_efficiency * truck_emission_factor / 1000., axis=1)
                logger.info("   Done with {}".format(f))

                ### Local transport emissions in importing country
            logger.info("Done looping through feeds")

            # Calculate cost of producing feed for local meat production
            local_cost_grain = np.nansum(
                biomass_dom * feats[['ADM0_A3']].merge(feedprices[['ADM0_A3'] + foddercrop_list], how="left").drop(
                    "ADM0_A3", axis=1).values, axis=1)
            # Calculate cost of producing grass
            grass_cost = np.take_along_axis(feats[[lu + '_cost' for lu in grass_cols]].values, grasslu[:, None], axis=1).flatten()

            # Calculate total cost
            local_cost = local_cost_grain + (grass_cost * (grass_area/grassarea2))

            # Number of trips to bring feed to port
            ntrips_feed_exp = np.nansum(biomass_exported, axis=1) / int(15) * 2
            ntrips_feed_exp = np.where(ntrips_feed_exp < 0, 0, ntrips_feed_exp)
            # Cost of sending feed to port
            feed_to_port_cost = ntrips_feed_exp * feats["distance_port"] * \
                                feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                         how='left',
                                                         left_on='ADM0_A3',
                                                         right_on='ADM0_A3')['Diesel'].values * fuel_efficiency / 1000.

            # Emissions from transporting feed to nearest port (tons)
            feed_to_port_emis = ntrips_feed_exp * feats[
                'distance_port'] * fuel_efficiency * truck_emission_factor / 1000.

            # Total meat production
            feats[l + '_meat'] = meat_abroad + local_meat

            # Total meat production from grain
            feats['grain_grain_meat'] = local_grain_meat + meat_abroad
            # Total meat production from grass
            feats['grain_grass_meat'] = grass_meat

            # Number of trips to markets
            ntrips_beef_mkt = feats[l + '_meat'].values / int(15) * 2
            ntrips_beef_mkt = np.where(ntrips_beef_mkt < 0, 0, ntrips_beef_mkt)

            beef_trans_cost = ntrips_beef_mkt * feats[['ADM0_A3']].merge(fuel_cost[[
                'ADM0_A3', 'Diesel']], how='left', left_on='ADM0_A3', right_on='ADM0_A3')['Diesel'].values * \
                              feats["accessibility"] * fuel_efficiency / 1000.

            # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
            beef_trans_emiss = ntrips_beef_mkt * feats[
                "accessibility"] * fuel_efficiency * truck_emission_factor / 1000.
            logger.info("Done calculating costs and emissions")

            feats[l + '_meth'] = methane_abroad + local_methane
            feats[l + '_manure'] = manure_abroad + local_manure
            feats[l + '_cost'] = local_cost
            feats[l + '_trans_cost'] = beef_trans_cost + feed_to_port_cost + exp_costs + trancost_partner_ls
            feats[l + '_trans_emiss'] = beef_trans_emiss + feed_to_port_emis + sea_emissions_ls + emissions_partner_ls

            cstock_grain = np.where(feats['newarea'].values == 1,
                                            feats['agb_spawn'].values * 3.67 * feats[l + '_area'].values,
                                            0)
            cstock_grass = np.take_along_axis(feats[[lu + '_cstock' for lu in grass_cols]].values, grasslu[:, None], axis=1).flatten()

            # 0 C stock for grain. For grass, take cstock from the best grass land use and multiply by the fraction of area
            feats[l + '_cstock'] = cstock_grass * (grass_area/grassarea2) + cstock_grain

            # Calculate change in below ground biomass for pasture/tree to grain ans and grain/tree to pasture
            bgb_change_grain = (((-0.59 * feats['pasture_area'] * feats['bgb_spawn']) + (
                    -0.42 * feats['tree_area'] * feats['bgb_spawn'])) * 3.67 * -1) * grain_area / \
                               feats['suitable_area']

            bgb_change_grass = ((0.19 * feats['crop_area'] * feats['bgb_spawn']) + (
                        0.08 * feats['tree_area'] * feats['bgb_spawn'])) * 3.67 * -1 * grass_area / \
                               feats['suitable_area']

            # Calculate total change in below ground biomass
            feats[l + '_bgb_change'] = eac(bgb_change_grain + bgb_change_grass, rate=0)

            logger.info("Done writing cropland columns")

            feats = weighted_score(feats, l, lam, optimisation_method)

            feats['grain_grassBM'] = feats['grain_grassBM'].fillna(0)
            feats['grain_grainBM'] = feats['grain_grainBM'].fillna(0)

            # Update score: Set score to 0 if either grass or grain is 0
            feats[l + '_score'] = np.where((feats['grain_grassBM'] == 0) | (feats['grain_grainBM'] == 0),
                np.nan, feats[l + '_score'].values)

            del beef_trans_emiss, feed_to_port_emis, sea_emissions_ls, emissions_partner_ls, \
                beef_trans_cost, feed_to_port_cost, exp_costs, trancost_partner_ls, local_cost, manure_abroad, \
                local_manure, methane_abroad, local_methane, meat_abroad, local_meat, ntrips_beef_mkt, ntrips_feed_exp, \
                meat, biomass_dom

        for l in ['stover_grass']:
            grasslu = np.nanargmin(np.ma.masked_array(feats[[g + '_score' for g in grass_cols]].values,
                                                      np.isnan(feats[[g + '_score' for g in
                                                                      grass_cols]].values)), axis=1)

            feats['stover_grass_grassBM'] = np.take_along_axis(feats[[lu + '_BM' for lu in grass_cols]].values,
                                                        grasslu[:, None], axis=1).flatten()

            stover_fraction = feats[['region']].merge(feed_compo[['region', 'stover']], how = 'left')['stover'].values
            stover_max = feats['stover_grass_grassBM'] * (stover_fraction / (1-stover_fraction))

            # Stover production (t) = crop production for other uses (t) * stover availability (%)

            potential_stover = feats.available_area.values / (feats.suitable_area.values - feats.net_fodder_area) * feats['stover_bm'].values

            # Adjust stover production based on what is needed
            # If potential is greater than maximum, adjust stover production by a ratio of max/potential
            # stov_adjustment = np.where(np.nansum(potential_stover, axis = 1) > stover_max, stover_max/np.nansum(potential_stover, axis = 1), 1.)
            stov_adjustment = np.where(potential_stover > stover_max, stover_max/potential_stover, 1.)

            stover_production = potential_stover * stov_adjustment

            # Stover energy ('000 MJ) = sum across rows of Stover production (t) * stover energy (MJ/kg dm)
            # stover_energy = np.nansum(stover_production * residue_energy.iloc[0].values[None, :], axis=1)
            stover_energy = feats.available_area.values / (feats.suitable_area.values - feats.net_fodder_area) * feats['stover_energy'].values * stov_adjustment

            feats['stover_grass_stoverBM'] = stover_production

            feats[l + '_BM'] = feats['stover_grass_stoverBM'] + feats['stover_grass_grassBM']
            # Subset meat table for mixed systems
            subset_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['group', 'glps', beef_yield,
                                                                                     'curr_methane', 'curr_manure']]
            # Meat production (t) = Stover energy ('000 MJ) * energy converion (kg/MJ) * dressing percentage
            meat_stover = stover_energy * feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
                                                         right_on=['group', 'glps'])[beef_yield].values * dressing

            # Update meat production after climate penalty
            meat_stover = np.sum(np.where(feats[months] < -1,
                                                 (meat_stover[:, None] - (meat_stover[:, None] * (
                                                             -0.0182 * feats[months] - 0.0182))) / 12.,
                                                 meat_stover[:, None] / 12.), axis=1)

            grass_meat = np.take_along_axis(feats[[lu + '_meat' for lu in grass_cols]].values,
                       grasslu[:, None], axis=1).flatten()

            feats[l + '_meat'] = meat_stover + grass_meat

            feats['stover_grass_grass_meat'] = grass_meat
            feats['stover_grass_stover_meat'] = meat_stover

            # Methane emissions (t CO2 eq) = Biomass consumed (t) * CH4 emissions (t CO2 eq/t)
            stover_methane = stover_production * \
                                 feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
                                                                right_on=['group', 'glps'])['curr_methane'].values
            feats[l + '_meth'] = stover_methane + np.take_along_axis(feats[[lu + '_meth' for lu in grass_cols]].values,
                                 grasslu[:, None], axis=1).flatten()

            # Manure N20 (t CO2 eq) = Biomass consumed (t) * N2O emissions (t CO2 eq/t)
            stover_manure = stover_production * \
                                 feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
                                                                right_on=['group', 'glps'])['curr_manure'].values
            feats[l + '_manure'] = stover_manure + np.take_along_axis(feats[[lu + '_manure' for lu in grass_cols]].values,
                                 grasslu[:, None], axis=1).flatten()

            # Trips to market
            ntrips_beef_mkt = feats[l + '_meat'].values / int(15) * 2
            ntrips_beef_mkt = np.where(ntrips_beef_mkt < 0, 0, ntrips_beef_mkt)

            # Transport emissions = number of trips to nearest market * distance to market (km) * fuel efficeincy (l/km) * Diesel cost (USD/l)
            feats[l + '_trans_cost'] = ntrips_beef_mkt * feats["accessibility"] * fuel_efficiency * feats[
                ['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']], how='left', left_on='ADM0_A3',
                                   right_on='ADM0_A3')['Diesel'].values / 1000.

            # Transport emissions = number of trips to nearest market * distance to market (km) * fuel efficeincy (l/km) * emissions factor (kg CO2/l) * kg/t conversion
            feats[l + '_trans_emiss'] = ntrips_beef_mkt * feats[
                "accessibility"] * fuel_efficiency * truck_emission_factor / 1000.


            for col in ['_cost', '_n2o', '_cstock', '_bgb_change']:
                feats[l + col] = np.take_along_axis(feats[[glu + col for glu in grass_cols]].values,
                                                        grasslu[:, None], axis=1).flatten()

            # For grazing, convert all area
            feats[l + '_area'] = feats['available_area'].values


            feats['stover_grass_grassBM'] = feats['stover_grass_grassBM'].fillna(0)
            feats['stover_grass_stoverBM'] = feats['stover_grass_stoverBM'].fillna(0)

            feats[l + '_meat'] = np.where((feats['stover_grass_grassBM'] == 0) | (feats['stover_grass_stoverBM'] == 0),
                np.nan, feats[l + '_meat'].values)

            feats = weighted_score(feats, l, lam, optimisation_method)

        for l in ['stover_grain']:
            # fodder_potential_yields = potential_yields[['climate_bin'] + [c for c in foddercrop_list]]
            fodder_yield_fraction = yield_fraction[['ADM0_A3'] + foddercrop_list]

            #### Local feed consumption ####
            grain_production = feats['available_area'].values[:, None] * \
                               feats[['ADM0_A3']].merge(foddercrop_area, how="left").drop('ADM0_A3', axis=1).values * feats[foddercrop_list].values *\
                                feats[['ADM0_A3']].merge(fodder_yield_fraction, how="left").drop('ADM0_A3', axis=1).values
                               # feats[['climate_bin']].merge(fodder_potential_yields, how="left").drop('climate_bin', axis=1).values *\


            # potential_stover = (feats[['diff_' + i for i in foddercrop_list]].values + grain_production) * crop_residues[foddercrop_list].iloc[0].values[None,
            #                                                                      :] * stover_removal

            potential_stover = feats.available_area.values / (feats.suitable_area.values - feats.net_fodder_area) * feats['stover_bm'].values

            stover_fraction = feats[['region']].merge(feed_compo[['region', 'stover']], how = 'left')['stover'].values
            stover_max = np.nansum(grain_production, axis = 1) * (stover_fraction / (1-stover_fraction))

            # Adjust stover production based on what is needed
            # If potential is greater than maximum, adjust stover production by a ratio of max/potential
            # stov_adjustment = np.where(np.nansum(potential_stover, axis = 1) > stover_max, stover_max/np.nansum(potential_stover, axis = 1), 1.)
            stov_adjustment = np.where(potential_stover > stover_max, stover_max/potential_stover, 1.)

            # stover_production = potential_stover * stov_adjustment[:,None]

            stover_production = potential_stover * stov_adjustment
            # feats['stover_grain_stoverBM'] = np.nansum(stover_production, axis = 1)

            grain_max = stover_production * (0.8 / (1-0.8))

            grain_adjustment = np.where(np.nansum(grain_production, axis = 1) > grain_max, grain_max/np.nansum(grain_production, axis = 1), 1.)
            grain_production = grain_production * grain_adjustment[:, None]

            feats['stover_grain_grainBM'] = np.nansum(grain_production, axis = 1)
            feats['stover_grain_stoverBM'] = stover_production
            feats[l + '_BM'] = feats['stover_grain_grainBM'] + feats['stover_grain_stoverBM']
            feats[l + '_area'] = feats['available_area'].values

            # Biomass consumed for domestic production (t) = actual production (t) x (1 - fraction exported feed)
            # biomass_dom = total_prod * grain_perc * (
            #         1 - feats[['ADM0_A3']].merge(percent_exported, how="left").drop('ADM0_A3', axis=1).values)

            # # Biomass consumed for domestic production (t) = actual production (t) x fraction exported feed
            biomass_exported = grain_production * feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3'] + foddercrop_list],
                                                                     how="left").drop('ADM0_A3', axis=1).values
            biomass_dom = grain_production * (1 - feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3'] + foddercrop_list],
                                                                     how="left").drop('ADM0_A3', axis=1).values)
            # Subset ME in conversion per region and climate
            subset_table = energy_conversion.loc[energy_conversion.feed == 'mixed'][['group', 'glps', beef_yield,
                                                                                     'curr_methane', 'curr_manure']]

            # stover_energy = feats['stover_energy'].values * stov_adjustment

            # Meat production (t) = sum across feeds (Domestic biomass (t) x ME in feed (MJ/kd DM)) x ME to beef conversion ratio * dressing (%)
            stover_meat = feats.available_area.values / (feats.suitable_area.values - feats.net_fodder_area) * feats['stover_energy'].values * stov_adjustment * \
                   feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
                                                  right_on=['group', 'glps'])[beef_yield].values * dressing

            local_grain_meat = np.nansum(biomass_dom * feed_energy[foddercrop_list].iloc[0].values[None, :], axis=1) * \
                   feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
                                                  right_on=['group', 'glps'])[beef_yield].values * dressing
            # meat = (np.nansum(biomass_dom * feed_energy[foddercrop_list].iloc[0].values[None, :], axis=1) + (
            #     np.nansum(stover_production * residue_energy[foddercrop_list].iloc[0].values[None, :], axis=1))) * \
            #        feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
            #                                       right_on=['group', 'glps'])[beef_yield].values * dressing
            local_grain_meat = np.where(local_grain_meat < 0, 0, local_grain_meat)
            stover_meat = np.where(stover_meat < 0, 0, stover_meat)

            # Update meat production after climate penalty
            stover_meat = np.sum(np.where(feats[months] < -1,
                                         (stover_meat[:, None] - (stover_meat[:, None] * (-0.0182 * feats[months] - 0.0182))) / 12.,
                                         stover_meat[:, None] / 12.), axis=1)
            local_grain_meat = np.sum(np.where(feats[months] < -1,
                                         (local_grain_meat[:, None] - (local_grain_meat[:, None] * (-0.0182 * feats[months] - 0.0182))) / 12.,
                                         local_grain_meat[:, None] / 12.), axis=1)
            local_meat = stover_meat + local_grain_meat

            # Calculate methane produced from local beef production (ton) = biomass consumed (ton) x biomass-methane conversion (ton/ton)
            local_methane = (np.nansum(grain_production, axis=1) + stover_production) * \
                            feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
                                                           right_on=['group', 'glps'])['curr_methane'].values

            # Calculate N2O from manure from energy consumed with coefficients (ton CO2eq) = biomass consumed (ton) * conversion factor (ton CO2eq/tom DM)
            local_manure = (np.nansum(grain_production, axis=1) + stover_production) * \
                            feats[['group', 'glps']].merge(subset_table, how='left', left_on=['group', 'glps'],
                                                           right_on=['group', 'glps'])['curr_manure'].values

            # Calculate nitrous N2O (ton) = Actual production (ton) x fertiliser requirement (kg) x crop_emission factors (% per thousand)
            # feats[l + '_n2o'] = np.nansum(grain_production * fertiliser_requirement['fertiliser'].values[None, :] * (
            #         crop_emissions_factors['factor'].values[None, :] / 100), axis=1)
            feats[l + '_n2o'] = np.nansum(feats[l + '_area'].values[:, None] * feats[['ADM0_A3']].merge(foddercrop_area, how="left").drop(
                'ADM0_A3', axis=1).values * feats[['ADM0_A3']].merge(crop_emissions[['ADM0_A3'] + foddercrop_list], how="left").drop('ADM0_A3', axis=1).values,
                      axis = 1)

            logger.info("Done with local meat production")

            ##### Exported feed #####
            # Create empty arrays to fill in
            meat_abroad = np.zeros_like(feats.ADM0_A3, dtype='float32')
            methane_abroad = np.zeros_like(feats.ADM0_A3, dtype='float32')
            manure_abroad = np.zeros_like(feats.ADM0_A3, dtype='float32')
            exp_costs = np.zeros_like(feats.ADM0_A3, dtype='float32')
            sea_emissions_ls = np.zeros_like(feats.ADM0_A3, dtype='float32')
            emissions_partner_ls = np.zeros_like(feats.ADM0_A3, dtype='float32')
            trancost_partner_ls = np.zeros_like(feats.ADM0_A3, dtype='float32')

            for f in foddercrop_list:  # Loop though feeds
                ### Meat produced abroad
                # Quantity of feed f exported
                if feed_option == "v1":
                    # Qty exported (t) = Suitable area (ha) * crop area fraction * crop yield (t/ha) * yield gap (%) * export fraction
                    qty_exported = ((feats[l + '_area'].values * feats[['ADM0_A3']].merge(
                        foddercrop_area[['ADM0_A3', f + '_area']], how="left")[f + '_area'].values * \
                                     # feats[['climate_bin']].merge(fodder_potential_yields[['climate_bin', f]],
                                     #                              how="left")[f + '_potential'].values * \
                        feats[f].values * feats[['ADM0_A3']].merge(fodder_yield_fraction, how="left")[f].values)) * \
                                   feats[['ADM0_A3']].merge([['ADM0_A3', f]], how="left")[f].values * grain_adjustment

                if feed_option == "v2":
                    # Qty exported (t) = (Suitable area (ha) * crop area fraction * crop yield (t/ha) * yield gap (%)) - production for other uses (t) * export fraction
                    qty_exported = ((feats[l + '_area'].values * \
                                     feats[['ADM0_A3']].merge(foddercrop_area[['ADM0_A3', f + '_area']], how="left")[
                                         f + '_area'].values * feats[f].values * feats[['ADM0_A3']].merge(fodder_yield_fraction, how="left")[f].values)

                                     # feats[['climate_bin']].merge(fodder_potential_yields[['climate_bin', f]],
                                     #                              how="left")[f].values * \
                                    # - feats['diff_' + f].values
                                    ) * feats[['ADM0_A3']].merge(percent_exported[['ADM0_A3', f]], how="left")[f].values * grain_adjustment

                qty_exported = np.where(qty_exported < 0, 0, qty_exported)

                # trade partners
                trade_partners = feats[['ADM0_A3']].merge(feedpartners.loc[feedpartners.crop == f], how='left').drop(
                    ['ADM0_A3', 'crop'], axis=1).values

                # Meat produced from exported feed (t) = Exported feed (t) * partner fraction (%) * energy in feed ('000 MJ/t) * energy conversion in partner country (t/'000 MJ) * dressing (%)
                meat_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * feed_energy[f].iloc[0] * partner_me['meat'].values[None,
                                                                                      :], axis=1) * dressing

                ### Methane emitted abroad (t CO2 eq) = Exported feed (t) * partner fraction (%) * methane emissions per biomass consumed (t/t)
                methane_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * partner_me["methane"].values[None, :], axis=1)

                ### N2O from manure emitted abroad (t CO2 eq) = Exported feed (t) * partner fraction (%) * N2O emissions per biomass consumed (t/t)
                manure_abroad += np.nansum(
                    qty_exported[:, None] * trade_partners * partner_me["manure"].values[None, :], axis=1)

                ### Export cost ('000 USD) = Exported feed (t) * partner fraction (%) * value of exporting crop c to partner p ('000 USD/t)
                exp_costs += np.nansum(qty_exported[:, None] * trade_partners * feats[['ADM0_A3']].merge(
                    expcosts.loc[expcosts.crop == f], how='left').drop(['ADM0_A3', 'crop'], axis=1).values, axis=1)

                ### Sea emissions (t CO2 eq) = Exported feed (t) * partner fraction (%) * sea distance from partner p (km) * sea emissions (kg CO2 eq/t-km) * kg-t conversion
                sea_emissions_ls += np.nansum(qty_exported[:, None] * trade_partners * feats[['ADM0_A3']].merge(
                    sea_dist, how='left').drop(['ADM0_A3'], axis=1).values * sea_emissions, axis=1) / 1000.

                ### Number of local transport cost in importing country
                ntrips_local_transp = qty_exported[:, None] * trade_partners / int(15) * 2

                ### Transport cost in partner country ('000 USD) = trips * accessibility to market in partner country (km) * fuel cost in partner country * fuel efficiency * USD-'000 USD conversion
                trancost_partner_ls += np.nansum(
                    ntrips_local_transp * exp_access['access'].values[None, :] * fuel_partner[
                                                                                     'Diesel'].values[None,
                                                                                 :] * fuel_efficiency / 1000., axis=1)

                ### Transport emissions in partner country (t CO2 eq) = trips * accessibility to market in partner country (km) *
                # fuel efficiency (l/km) * truck emission factor (kg CO2 eq/l) * kg-ton conversion
                emissions_partner_ls += np.nansum(
                    ntrips_local_transp * exp_access['access'].values[None,
                                          :] * fuel_efficiency * truck_emission_factor / 1000., axis=1)
                logger.info("   Done with {}".format(f))

                ### Local transport emissions in importing country
            logger.info("Done looping through feeds")

            local_cost = np.nansum(
                grain_production * feats[['ADM0_A3']].merge(feedprices[['ADM0_A3'] + foddercrop_list], how="left").drop("ADM0_A3", axis=1).values, axis=1)

                       # Number of trips to markets
            ntrips_beef_mkt = local_meat / int(15) * 2
            ntrips_beef_mkt = np.where(ntrips_beef_mkt < 0, 0, ntrips_beef_mkt)

            beef_trans_cost = ntrips_beef_mkt * feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                                         how='left',
                                                                         left_on='ADM0_A3',
                                                                         right_on='ADM0_A3')['Diesel'].values * \
                              feats["accessibility"] * fuel_efficiency / 1000.

            # Transport emissions: number of trips * emissions per trip (tons CO2 eq)
            beef_trans_emiss = ntrips_beef_mkt * feats[
                "accessibility"] * fuel_efficiency * truck_emission_factor / 1000.
            logger.info("Done calculating costs and emissions")

            feats[l + '_meat'] = local_meat + meat_abroad

            feats['stover_grain_grain_meat'] = meat_abroad + local_grain_meat
            feats['stover_grain_stover_meat'] = stover_meat

            feats[l + '_meth'] = local_methane + methane_abroad
            feats[l + '_manure'] = local_manure + manure_abroad
            feats[l + '_cost'] = local_cost

            # Number of trips to bring feed to port
            ntrips_feed_exp = np.nansum(biomass_exported, axis=1) / int(15) * 2
            ntrips_feed_exp = np.where(ntrips_feed_exp < 0, 0, ntrips_feed_exp)
            # Cost of sending feed to port
            feed_to_port_cost = ntrips_feed_exp * feats["distance_port"] * \
                                feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                         how='left',
                                                         left_on='ADM0_A3',
                                                         right_on='ADM0_A3')['Diesel'].values * fuel_efficiency / 1000.

            # Total cost of exporting feed
            # Emissions from transporting feed to nearest port (tons)
            feed_to_port_emis = ntrips_feed_exp * feats[
                'distance_port'] * fuel_efficiency * truck_emission_factor / 1000.

            feats[l + '_trans_cost'] = beef_trans_cost + feed_to_port_cost + exp_costs + trancost_partner_ls
            feats[l + '_trans_emiss'] = beef_trans_emiss + feed_to_port_emis + sea_emissions_ls + emissions_partner_ls

            feats[l + '_cstock'] = np.where(feats['newarea'].values == 1,
                                            feats['agb_spawn'].values * 3.67 * feats[l + '_area'].values,
                                            0)

            bgb_change_grain = (((-0.59 * feats['pasture_area'] * feats['bgb_spawn']) + (
                    -0.42 * feats['tree_area'] * feats['bgb_spawn'])) * 3.67 * -1) * (feats[l + '_area'] / \
                               feats['suitable_area'])
            feats[l + '_bgb_change'] = eac(bgb_change_grain, rate=0)

            feats['stover_grain_grainBM'] = feats['stover_grain_grainBM'].fillna(0)
            feats['stover_grain_stoverBM'] = feats['stover_grain_stoverBM'].fillna(0)

            feats = weighted_score(feats, l, lam, optimisation_method)

            feats[l + '_score'] = np.where((feats['stover_grain_grainBM'] == 0) | (feats['stover_grain_stoverBM'] == 0),
                                          np.nan, feats[l + '_score'].values)

    else:
        logger.inf('Feed option {} not in choices'.format(feed_option))
    # Drop monthly temperature and other crop uses columns
    # feats = feats.drop(months + ['diff_' + crop for crop in crop_list], axis=1)
    # feats = feats.drop(months, axis=1)

    # Do not consider production lower than 1 ton
    feats[[l + '_meat' for l in landuses]] = np.where(feats[[l + '_meat' for l in landuses]] < 1, 0, feats[[l + '_meat' for l in landuses]])

    # Only keep cells where at least 1 feed option produces meat
    feats = feats.loc[feats[[l + '_meat' for l in landuses]].sum(axis=1) > 0]

    # Drop rows where no land use has a score (all NAs)
    feats = feats.dropna(how='all', subset=[l + '_score' for l in landuses])

    # Select lowest score
    feats['best_score'] = np.nanmin(feats[[l + '_score' for l in landuses]].values, axis=1)

    try:
        # Select position (land use) of lowest score
        feats['bestlu'] = np.nanargmin(feats[[l + '_score' for l in landuses]].values, axis=1)
    except:
        # If there is no best land use, export dataframe
        feats.loc[feats.best_score.isna()].drop('geometry', axis=1).to_csv("nadf.csv", index=False)

    # del list_scores, allArrays

    # Create a new column for all variables in new_colnames that selects the value of the optimal land use
    # for i in new_colnames:
    #     feats[i] = np.take_along_axis(feats[[l + new_colnames[i] for l in landuses]].values,
    #                                   feats['bestlu'].values[:, None], axis=1)
    for cname in ['production']:
        feats[cname] = np.take_along_axis(feats[[lu + new_colnames[cname] for lu in landuses]].values,
                                          feats['bestlu'].values[:, None], axis=1).flatten()

    return feats

def trade(feats, optimisation_method, lam, feed_option, landuses):
    """
    Function to update score based on trade costs and emissions

    Arguments:
    feats (pandas dataframe) -> Main dataframe
    optimisation_method (str)-> Method for optimisation ('weighted_sum' or 'carbon_price')
    lam (float)-> Lambda weight ([0,1])
    feed_option (str)-> folder where the output file is exported

    Output: returns a gridded dataframe with updated score
    """
    if feed_option == 'v3':
        landuses = ['grazing', 'mixed']

    for l in landuses:

        # Calculate transport trips to export meat
        ntrips = (feats[l + '_meat'] / int(15) + 1) * 2

        # Calculate transport cost to nearest port
        feats[l + '_trans_cost'] = ntrips * feats["distance_port"] * \
                                   feats[['ADM0_A3']].merge(fuel_cost[['ADM0_A3', 'Diesel']],
                                                            how='left', left_on='ADM0_A3',
                                                            right_on='ADM0_A3')['Diesel'].values * fuel_efficiency/1000.

        # Calculate transport costs as a function of quantity traded
        feats[l + '_exp_costs'] = feats[l + '_meat'] * feats[['ADM0_A3']].merge(sea_t_costs[['ADM0_A3', 'tcost']],
                                                                                how = 'left')['tcost'].values

        # Transport emissions to port
        feats[l + '_trans_emiss'] = ntrips * feats["distance_port"] * fuel_efficiency * truck_emission_factor / 1000.

        # Transport emissions by sea
        feats[l + '_exp_emiss'] = feats[['ADM0_A3']].merge(sea_distances[['ADM0_A3', 'ave_distance']], how='left')['ave_distance'].values * feats[l + '_meat'] * sea_emissions / 1000.

        # Update total cost ('000 USD)
        feats[l + '_tot_cost'] = feats[l + '_est_cost'] +  feats[l + '_cost'] + feats[l + '_trans_cost'] +\
                                 feats[l + '_opp_cost'] + feats[l + '_exp_costs'] - feats['aff_cost']
        # + feats[l + '_compensation'])

        # Update annual emissions (t CO2 eq)
        flow = feats[l + '_n2o'] + feats[l + '_meth'] + feats[l + '_manure'] + feats[l + '_trans_emiss'] + feats[l + '_exp_emiss'] + feats[l + '_process_energy']

        # Update total emissions (t CO2 eq)
        feats[l + '_ghg'] = feats[l + '_agb_change'] + feats[l + '_bgb_change'] + flow - feats['opp_aff'] - feats[
            'opp_soc']

    # Drop rows where columns are all nas
    feats = feats.dropna(how='all', subset=[lu + '_score' for lu in landuses])

    # make sure that dataframe is not empty
    if feats.shape[0] > 0:
        if optimisation_method == 'carbon_price':
            lam = lam/1000.

        rel_ghg = np.where(feats[l + '_meat'] < 1, np.NaN, feats[l + '_ghg'] / (feats[l + '_meat']))
        # Calculate relative Cost (GHG/meat)
        rel_cost = np.where(feats[l + '_meat'] < 1, np.NaN,
                            feats[l + '_tot_cost'] / (feats[l + '_meat']))

        feats[l + '_score'] = (rel_ghg * (1 - lam)) + (rel_cost * lam)

        feats['best_score'] = np.nanmin(feats[[l + '_score' for l in landuses]].values, axis=1)

        try:
            # Select position (land use) of lowest score
            feats['bestlu'] = np.nanargmin(feats[[l + '_score' for l in landuses]].values, axis=1)
        except:
            # If there is no best land use, export dataframe
            feats.loc[feats.best_score.isna()].drop('geometry', axis=1).to_csv("nadf.csv", index=False)

        # Write new columns according to optimal land use
        for cname in ['production']:
            feats[cname] = np.take_along_axis(feats[[lu + new_colnames[cname] for lu in landuses]].values,
                                              feats['bestlu'].values[:, None], axis=1).flatten()
        return feats

def export_raster(grid, resolution, export_column, export_folder, constraint, crop_yield, beef_yield,
                  lam, demand_scenario, feed_option, aff_scenario):

    """
    Function to rasterize columns of a dataframe

    Arguments:
    grid (pandas dataframe)-> Dataframe to rasterize
    resolution (float)-> Resolution at which to rasterize
    export_column (list)-> list of columns to rasterize

    export_folder (str)-> folder where the output file is exported
    constraint (str)-> Spatial constraint for beef production ('global', 'country', or 'subsistence')
    crop_yield (int)-> Scenario of crop yield (0 = current, 1 = no yield gap)
    beef_yield (str)-> Scenario of beef yield ('me_to_meat' = current, 'max_yield' = no yield gap)
    lam (float)-> Lambda weight ([0,1])
    demand_scenario (str)-> Scenario of beef demand ('SSP1-NoCC2010' or 'SSP1-NoCC2050')
    feed_option (str)-> folder where the output file is exported

    Output: Writes the grid as GPKG file
    """
    bounds = [-180,-90, 180, 90]
    resolution = float(resolution)
    width = abs(int((bounds[2] - bounds[0]) / resolution))
    heigth = abs(int((bounds[3] - bounds[1]) / resolution))
    out_shape = (heigth, width)
    # grid['bestlu'] = np.array(grid['bestlu'], dtype='uint8')
    for i in export_column:

        dt = grid[i].dtype

        print("Type of array: {}, type of file: {}".format(grid[i].dtype, dt))
        meta = {'driver': 'GTiff',
            'dtype': dt,
            'nodata': 0,
            'width': width,
            'height': heigth,
            'count': 1,
            'crs': {'init': 'epsg:4326'},
            'transform': Affine(resolution, 0.0, bounds[0],
                                0.0, -resolution, bounds[3]),
            'compress': 'lzw',
            }
        # for m in meta: print(m, meta[m])
        out_fn = export_folder + '/' + constraint + "_" + str(crop_yield) + '_' + beef_yield + '_' + str(lam) + '_' + demand_scenario + '_' + feed_option + '_' + aff_scenario + ".tif"
          
        with rasterio.open(out_fn, 'w', **meta) as out:
            # Create a generator for geom and value pairs
            grid_cell = ((geom, value) for geom, value in zip(grid.geometry, grid[i]))

            burned = features.rasterize(shapes=grid_cell, fill=0, out_shape=out_shape, dtype = dt,
                                        transform=Affine(resolution, 0.0, bounds[0],
                                                         0.0, -resolution, bounds[3]))
            print("Burned value dtype: {}".format(burned.dtype))
            out.write_band(1, burned)

def main(export_folder ='.', optimisation_method= 'weighted_sum', lam = 0.5, demand_scenario = 'Demand',
         crop_yield = 0, beef_yield ='me_to_meat', constraint ='global', aff_scenario = "no_aff", feed_option ='v1', trade_scenario ='trade'):
    """
    Main function that optimises beef production for a given location and resolution, using a given number of cores.
    
    Arguments:
    export_folder (str)-> folder where the output file is exported
    optimisation_method (str)-> Method for optimisation ('weighted_sum' or 'carbon_price')
    lam (float)-> Lambda weight ([0,1])
    demand_scenario (str)-> Scenario of beef demand ('SSP1-NoCC2010' or 'SSP1-NoCC2050')
    crop_yield (int)-> Scenario of crop yield (0 = current, 1 = no yield gap)
    beef_yield (str)-> Scenario of beef yield ('me_to_meat' = current, 'max_yield' = no yield gap)
    constraint (str)-> Spatial constraint for beef production ('global', 'country', or 'subsistence')
    feed_option (str)-> folder where the output file is exported
    trade_scenario (str)-> Trade scenario (if 'trade', apply trade based on country demand)

    Output: Writes the grid as GPKG file
    """

    LOG_FORMAT = "%(asctime)s - %(message)s"
    try:
        logging.basicConfig(
            # filename="/home/uqachare/model_file/logs_opt/opt_" + constraint + "_" + str(crop_yield) + "_" + me_to_meat + "_" + str(lam) + '_' + dem +".log",
            filename="/home/uqachare/model_file/test_" + feed_option + ".log",
            level=logging.INFO,
            format=LOG_FORMAT,
            filemode='w')
    except:
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            filemode='w')
    logger = logging.getLogger()

    logger.info("Start loading grid")

    import sqlite3
    conn = sqlite3.connect("grid.gpkg")
    grid = pd.read_sql_query("SELECT * FROM grid", conn)
    conn.close()

    # grid = gpd.read_file("grid.gpkg")
    # grid = grid.loc[grid.ADM0_A3 == 'BEL']
    # grid = grid.reset_index(drop=True)

    logger.info("Done loading grid, memory usage of grid = {}".format(grid.memory_usage().sum()*1e-6))

    dt_dict = {np.dtype('float64'): np.dtype('float32'),
               np.dtype('int64'): np.dtype('int32')}

    datatypes = pd.DataFrame({'dtypes': grid.dtypes,
                              'newtypes': [dt_dict[dt] if dt in dt_dict else dt for dt in grid.dtypes]})

    for id in datatypes.index:
        if id in ['glps', 'climate_bin', 'ecoregions']:
            grid[id] = grid[id].fillna(0).astype(np.dtype('int8'))
        else:
            grid[id] = grid[id].astype(datatypes.loc[id, 'newtypes'])

    grid['glps'] = grid['glps'].astype('int8')
    logger.info("End changing datatypes, memory usage of grid = {}".format(grid.memory_usage().sum()*1e-6))

    for i in ['soilmoisture', "gdd",  'ls_opp_cost', 'agri_opp_cost', 'est_area']:
        if i in grid.columns:
            grid = grid.drop(i, axis = 1)

    logger.info("Simulation start")
    logger.info('Me_to_meat scanerio: {}'.format(beef_yield))
    logger.info('Weight scenario: {}'.format(lam))
    logger.info('Feed option scenario: {}'.format(feed_option))

    # Set amount of beef to be produced based on the chosen location

    logger.info('Constraint: {}'.format(constraint))
    if constraint == 'subsistence':
        logger.info('Shape of all grid: {}'.format(grid.shape))
        pastoral = grid.loc[grid.beef_gs > 0]
        logger.info('Shape of pastoral grid: {}'.format(pastoral.shape))
        grid = grid.loc[~(grid.beef_gs > 0)]
        logger.info('Shape of grid without pastoral: {}'.format(grid.shape))

        pastoral['production'] = pastoral['beef_gs'] * 0.01 * pastoral.suitable_area * 1e-3

        demand = beef_demand[demand_scenario].sum()
        logger.info('Total demand: {}'.format(demand))
        demand = demand - pastoral['production'].sum()
        logger.info('Non-pastoral demand: {}'.format(demand))
        logger.info('Pastoral production: {}'.format(pastoral['production'].sum()))
    else:
        demand = beef_demand[demand_scenario].sum()
        # demand = 170000

    logger.info('Demand: {}'.format(demand))

    # Adjust other uses for future demand  Proportion of demand increase
    beef_demand['dem_increase'] = beef_demand[demand_scenario]/beef_demand['SSP1-NoCC2010']

    # logger.info('New demand for other uses before: {}'.format(grid[['diff_maize']].loc[grid.diff_maize > 0].head()))
    # other_uses = grid[['ADM0_A3']+['diff_' + i for i in crop_list]].merge(beef_demand[['ADM0_A3', 'dem_increase']])

    other_uses = grid[['ADM0_A3']].merge(beef_demand[['ADM0_A3', 'dem_increase']], how = 'left')['dem_increase'].values
    # grid[['diff_' + i for i in crop_list]] = grid[['diff_' + i for i in crop_list]].values * other_uses[:, None]
    grid['net_fodder_area'] = grid['net_fodder_area'].values * other_uses
    grid['stover_bm'] = grid['stover_bm'].values * other_uses
    grid['stover_energy'] = grid['stover_energy'].values * other_uses

    del other_uses

    landuses = grass_cols + ['grass_grain', 'stover_grass', 'stover_grain']  # landuses to include in the simulation

    # Set id to cell to merge back later
    grid['cell_id'] = range(1, grid.shape[0] + 1)

    #
    try:
        geom_df = grid[['geometry', 'cell_id']]
        grid = grid.drop('geometry', axis = 1)
    except:
        logger.info('No "geometry" column')
        geom_df = grid[['geom', 'cell_id']]
        grid = grid.drop('geom', axis = 1)

    grid['newarea'] = 1

    # Create dataframe of cells with current beef
    current = grid.loc[(grid.current_cropping + grid.current_grazing) > 0]
    # Create dataframe of cells with current beef
    current['newarea'] = 0

    logger.info('Grid shape before concat: {}'.format(grid.shape[0]))

    # Merge cells with beef with expansion cells
    grid = pd.concat([grid, current])
    grid['id'] = range(0, grid.shape[0])

    logger.info('Grid shape after concat: {}'.format(grid.shape[0]))

    grid = scoring(grid, optimisation_method, crop_yield, lam, beef_yield, aff_scenario, logger, feed_option, landuses)

    # Select only cells columns of meat production, emissions, costs and country column to export
    cols = []
    for l in landuses:
        for i in ['_meat', '_ghg', '_tot_cost', '_exp_emiss', '_exp_costs']:
            cols.append(l+i)
    cols.append('ADM0_A3')

    grid[cols + ['ADM0_A3', 'id', 'opp_nataff', 'opp_manaff', 'affor_cost', 'opport_soc']].to_csv('score_init.csv', index = False)


def parallelise(export_folder, optimisation_method, job_nmr, feed_option):

    # Loop through all scenarios to create a dictionary with scenarios and scenario id
    index = 1
    scenarios = {}
    for spat_cons in ['global', 'country']:
        for j in ['curr', 'max']:
            for k in [0, 1]:
                for d in ['SSP1-NoCC2010']:
                    for a in ['noaff', 'nataff', 'manaff']:
                        scenarios[index] = [spat_cons, j, k, d, a]
                        index += 1

    main(export_folder, optimisation_method,  # Optimisation method (weighted sum vs carbon price)
                                             0,  # Weight
                                             scenarios[job_nmr][3],  # Demand
                                             scenarios[job_nmr][2],  # Crop yield
                                             scenarios[job_nmr][1],  # Beef yield
                                             scenarios[job_nmr][0],  # Spatial constraint
                                             scenarios[job_nmr][4],  # Afforestation scenario
                                             feed_option,  # Feed option
                                             )

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('export_folder', help='Name of exported file')
    argparser.add_argument('optimisation_method', help='Which scenario of optimisation to run ("weighted_sum", "carbon_price")')
    argparser.add_argument('job_nmr', help="Job number referring to scenario", type=int)
    argparser.add_argument('feed_option', help="Options for calculating grain: v1 -> convert all cell to grain, v2 -> convert the difference between attainable yield and production for non-beef uses")

    args = argparser.parse_args()
    export_folder = args.export_folder
    optimisation_method = args.optimisation_method
    job_nmr = args.job_nmr
    feed_option = args.feed_option

    parallelise(export_folder, optimisation_method, job_nmr, feed_option)