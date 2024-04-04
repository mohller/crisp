import pandas as pnd
import sys
sys.path.append('./')

def nuclear_data_parser(filename=None):
    """Returns a pandas DataFrame which contains the information in the 
    file a txt file available in https://www.anl.gov/phy/atomic-mass-data-resources.
    """
    if filename is None:
        filename = 'nubase2016.txt'

    cut_idcs = [(  0,   3),
                (  4,   9),
                ( 11,  17), # first 3
                ( 18,  29),
                ( 29,  38),
                ( 38,  48), # first 6
                ( 48,  58),
                ( 58,  61),
                ( 61,  69),
                ( 69,  71), # - 10, units
                ( 72,  78),
                ( 79,  93),
                ( 93,  95),
                ( 96, 105),
                (105, 109),
                (109, -1)]
    col_names = ['A', 'isomer_id', 'symbol', 'mass_excess_keV', 'unc_mass_excess',
                'Eexc', 'unc_Eexc', 'decay?', 'half_life', 'half_life_units',       
                'unc_half_life', 'parity', 'spin', 'references', 'year', 'decays']
    
    table = pnd.read_fwf(filename, names=col_names, colspecs=cut_idcs, comments='#')

    # retouching table
    Z_column = pnd.to_numeric(table['isomer_id'].apply(lambda s: s[:3]))
    table.insert(1, 'Z', Z_column)

    table['half_life'] = table['half_life'].map(str)
    table['half_life'] = table['half_life'].map(lambda val: val.replace('stbl', 'inf'))

    faulty_rows = []
    rows = []
    for k, val in enumerate(table['half_life']):
        try:
            float(val)
        except:
            faulty_rows.append(k)
            rows.append(table.loc[k])
            
    for k in faulty_rows:
        table.drop(k, inplace=True)

    table['half_life'] = pnd.to_numeric(table['half_life'], errors='coerce')

    # cleaning up column excitation energy 
    series = table['Eexc'].map(str)
    series = series.map(lambda string: string.split('#')[0])
    series[series == 'nan'] = 0
    series[series == 'non-exist'] = 0
    table['Eexc'] = pnd.to_numeric(series)

    # cleaning up column excitation energy 
    series = table['mass_excess_keV'].map(str)
    series = series.map(lambda string: string.split('#')[0])
    series[series == 'nan'] = 0
    table['mass_excess_keV'] = pnd.to_numeric(series)

    def to_secs(units):
        time_factors = {
                        ''    :  1.,
                        'Ty'    :  1e12*365*24*3600.,
                        'Py'    :  1e15*365*24*3600.,
                        'as'    :  1e-18,
                        'Gy'    :  1e9*365*24*3600.,
                        'zs'    :  1e-21,
                        'ps'    :  1e-12,
                        'Yy'    :  1e24*365*24*3600.,
                        'ns'    :  1e-9,
                        'fs'    :  1e-15,
                        'ys'    :  1e-24,
                        'My'    :  1e6*365*24*3600.,
                        'd'    :  24*3600.,
                        'h'    :  3600,
                        'm'    :  60.,
                        'us'    :  1e-6,
                        's'    :  1,
                        'Ey'    :  1e18*365*24*3600.,
                        'Zy'    :  1e21*365*24*3600.,
                        'ms'    :  1e-3,
                        'y'    :  365*24*3600.,
                        'ky'    :  1e3*365*24*3600.
                        }

        if units in time_factors:
            return time_factors[units]
        else:
            return units 

    table['half_life_units'] = table['half_life_units'].map(str).map(to_secs)
    table['half_life_units'] = pnd.to_numeric(table['half_life_units'], errors='coerce')

    return table