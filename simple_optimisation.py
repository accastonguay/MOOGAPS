import pandas as pd
import numpy as np
import logging
import multiprocessing
import time

beef_production = pd.read_csv("./beef_production.csv")  # Load country-level beef demand

def main(spat_const):
    LOG_FORMAT = "%(asctime)s - %(message)s"
    try:
        logging.basicConfig(
            # filename="/home/uqachare/model_file/logs_opt/opt_" + constraint + "_" + str(crop_yield) + "_" + me_to_meat + "_" + str(lam) + '_' + dem +".log",
            filename="/home/uqachare/model_file/simpleopt_" + str(spat_const) + ".log",
            level=logging.INFO,
            format=LOG_FORMAT,
            filemode='w')
    except:
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            filemode='w')
    logger = logging.getLogger()
    grid = pd.read_csv('./score_init.csv')
    # grid = grid.loc[grid.ADM0_A3 == 'NZL']

    demand = 69477745
    # demand = 673218

    grass_cols = []

    for i in ["0250", "0375", "0500"]:
        for n in ["000", "050", "200"]:
            grass_cols.append("grass_" + i + "_N" + n)

    landuses = grass_cols + ['grass_grain', 'stover_grass', 'stover_grain']

    #--------------

    for w in range(0,11,1):
        weight = w/10.

        for l in landuses:
            rel_ghg = np.where(grid[l + '_meat'] < 1, np.NaN, grid[l + '_ghg'] / (grid[l + '_meat']))
            rel_cost = np.where(grid[l + '_meat'] < 1, np.NaN,
                                             grid[l + '_tot_cost'] / (grid[l + '_meat']))
            grid[l + '_score'] = (rel_ghg * (1 - weight)) + (rel_cost * weight)

        grid['score'] = np.nanmin(grid[[l + '_score' for l in landuses]].values, axis=1)

        try:
            grid['lu'] = np.nanargmin(grid[[l + '_score' for l in landuses]].values, axis=1)
        except:
            print(grid.loc[grid.score.isna()])

        grid['beef'] = np.take_along_axis(grid[[lu + '_meat' for lu in landuses]].values,
                                          grid['lu'].values[:, None], axis=1).flatten()
        grid['emissions'] = np.take_along_axis(grid[[lu + '_ghg' for lu in landuses]].values,
                                               grid['lu'].values[:, None], axis=1).flatten()
        grid['costs'] = np.take_along_axis(grid[[lu + '_tot_cost' for lu in landuses]].values,
                                           grid['lu'].values[:, None], axis=1).flatten()

        grid = grid.sort_values('score')

        logger.info('Beef at weight {}: {}'.format(weight, grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'beef'].sum() * 1e-6))
        logger.info('Emissions: {}'.format(grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'emissions'].sum() * 1e-6))
        logger.info('Costs: {}'.format(grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0), 'costs'].sum() * 1e-6))
        logger.info('Number of converted cells: {}'.format(grid.loc[(demand + grid['beef'] - grid['beef'].cumsum() > 0)].shape[0]))

        total = pd.DataFrame({'beef':[grid.loc[(demand + grid['beef'] > grid['beef'].cumsum()), 'beef'].sum()],
                              'costs':[grid.loc[(demand + grid['beef'] > grid['beef'].cumsum()), 'costs'].sum()],
                              'emissions':[grid.loc[(demand + grid['beef'] > grid['beef'].cumsum()), 'emissions'].sum()],
                              'weight': [weight/10.],
                              'cells':[grid.loc[(demand + grid['beef'] > grid['beef'].cumsum()), 'emissions'].shape[0]]})


    if spat_const == 'country':
        total = pd.DataFrame()

        for w in range(0, 11, 1):
            weight = w / 10.

            for l in landuses:
                rel_ghg = np.where(grid[l + '_meat'] < 1, np.NaN, grid[l + '_ghg'] / (grid[l + '_meat']))
                rel_cost = np.where(grid[l + '_meat'] < 1, np.NaN,
                                    grid[l + '_tot_cost'] / (grid[l + '_meat']))
                grid[l + '_score'] = (rel_ghg * (1 - weight)) + (rel_cost * weight)

            grid['score'] = np.nanmin(grid[[l + '_score' for l in landuses]].values, axis=1)

            try:
                grid['lu'] = np.nanargmin(grid[[l + '_score' for l in landuses]].values, axis=1)
            except:
                print(grid.loc[grid.score.isna()])

            grid['beef'] = np.take_along_axis(grid[[lu + '_meat' for lu in landuses]].values,
                                              grid['lu'].values[:, None], axis=1).flatten()
            grid['emissions'] = np.take_along_axis(grid[[lu + '_ghg' for lu in landuses]].values,
                                                   grid['lu'].values[:, None], axis=1).flatten()
            grid['costs'] = np.take_along_axis(grid[[lu + '_tot_cost' for lu in landuses]].values,
                                               grid['lu'].values[:, None], axis=1).flatten()

            grid = grid.sort_values('score')

            allcountries = pd.DataFrame()
            for country in beef_production.Code:
                demand = beef_production.loc[beef_production.Code == country, 'Value'].iloc[0]
                country_df = grid.loc[grid.ADM0_A3 == country]
                country_total = pd.DataFrame({'beef': [country_df.loc[(demand + country_df['beef'] > country_df['beef'].cumsum()), 'beef'].sum()],
                                  'costs': [country_df.loc[(demand + country_df['beef'] > country_df['beef'].cumsum()), 'costs'].sum()],
                                  'emissions': [
                                      country_df.loc[(demand + country_df['beef'] > country_df['beef'].cumsum()), 'emissions'].sum()]})
                allcountries = pd.concat([allcountries, country_total])

            allcountries_total = pd.DataFrame(
                {'beef': [allcountries.beef.sum()],
                 'costs': [allcountries.costs.sum()],
                 'emissions': [allcountries.emissions.sum()],
                 'weight': [weight]})

            total = pd.concat([total, allcountries_total])


        total.to_csv('./results_' + str(spat_const) + '.csv', index=False)


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('spat_const')

    args = argparser.parse_args()
    spat_const = args.spat_const

    main(spat_const)
