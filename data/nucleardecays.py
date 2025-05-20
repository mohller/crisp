from numpy import nan, inf, log, isclose, logical_and
import re
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

class NuclearDataTable():
    def __init__(self, filename=None):
        self.table = nuclear_data_parser(filename)

    def get_no_isomers_table(self):
        """Returns the table excluding the isomer states
        """
        noW_table = self.table[self.table['isomer_id'].apply(lambda v: 'W' not in v)]
        table_no_isom = noW_table[noW_table['Z'] * 10 == noW_table['isomer_id'].apply(int)]

        return table_no_isom

    def prepare_decay_table(self):
        """Based on the output of nuclear_data_parser returns
        a table containing the children and branching ratio per
        decay channel for a range of nuclei. 
        IMP!!! Only up to A=56 due to remaining parsing errors!
        """
        table = self.get_no_isomers_table()
        unstable = table[table['half_life'] < inf]

        expression = r'\s*(?P<decays>[\+\-A-Za-z\d]+)[=<>~\sLE]+(?P<value>[\.\d\?#\s\+\-eE]+[\(\)\[\]\w,\d=egsm\.]*);*'
        # TODO: the reggex above still fails for one or two cases with ',e+' in the string
        reg = re.compile(expression)

        decay_dict = {}
        for Z, A, lab, tau, units in unstable[['Z', 'A', 'decays', 'half_life', 'half_life_units']].values:
            if (lab is not nan) and reg.findall(lab):
                decay_dict[A*100 + Z] = {}
                decay_dict[A*100 + Z]['decay_time'] = tau * units / log(2) # decay time seconds
                decay_list = [val for val in reg.split(lab) if val]
                decay_dict[A*100 + Z]['channels'] = dict([(dec, val) for dec, val in zip(decay_list[::2], decay_list[1::2])])

        regvals = r'[\d.e\+\-]+|[\d.e\+\-]+\s[\d.]+|\?' # expression to capture possible branching values
        regdecs = re.compile(r'(?P<num>\d*)(?P<decay>p|e\+|n|d|t|A|EC|B[\+\-])') # expression to capture decays

        nucid = {
            'p'  : 101,
            'n'  : 100,
            'd'  : 201,
            't'  : 301,
            'A'  : 402,
            'e+' : 1,
            'EC' : 1,
            'B+' : 1,
            'B-' : -1,
        }
        
        new_decay_dict = {}
        for key, decaydata in decay_dict.items():
            new_decay_dict[key] = decaydata
            channels = []
            for dlab, val in decaydata['channels'].items():
                daughters = []
                for n, p in regdecs.findall(dlab):
                    if n:
                        daughters += int(n) * [nucid[p],]
                    else:
                        daughters.append(nucid[p])
                
                if daughters == []:
                    continue
                    print('Error empty daughters!! regex failed on string:', dlab)

                first_val = re.compile(regvals).findall(val)[0]
                if first_val != '?':
                    first_val = float(first_val) / 100
                else:
                    first_val = 1.

                channel = [first_val, ] + daughters
                channels.append(channel)
            
            new_decay_dict[key]['channels'] = channels

        # Removing repeating beta decays 
        corrected = {}
        for nuc, decaydata in new_decay_dict.items():
            chans = decaydata['channels']
            corrected[nuc] = decaydata

            if chans[0][1] in [1, -1]:
                indices = [idx for idx, chan in enumerate(chans) if chans[0][1] in chan]

                if len(indices) == 1:
                    continue
    
                tot = sum([chans[idx][0] for idx in indices])

                for idx in indices:
                    chans[idx][0] /= tot 

            corrected[nuc]['channels'] = chans

        # correcting decays with incorrect branching sum
        for nuc in [601, 4526, 4828, 5430]:
            tot = sum([chan[0] for chan in corrected[nuc]['channels']])
            
            for idx, _ in enumerate(corrected[nuc]['channels']):
                corrected[nuc]['channels'][idx][0] /= tot

        # crosschecking that branchings add up to unity
        for key, decaydata in corrected.items():
            chans = decaydata['channels']
            if not isclose(sum([ch[0] for ch in chans]), 1):
                print('!!! Problem found in branching:', key, chans, sum([ch[0] for ch in chans]))

        return corrected