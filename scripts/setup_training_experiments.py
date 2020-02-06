#!/usr/bin/env python
import os, sys
import itertools

'''
python /home/ubuntu/speech2speech/scripts/train_model.py --verbose --debug\
    --base-dir "/home/ubuntu/speech2speech/test_num_embeddings256_spectrogram_db_lr1em4"\
    --data-dir "/home/ubuntu/speech2speech/data/raw/VCTK-Corpus"\
    --spectrogram-dir "/home/ubuntu/speech2speech/data/interim/spectogram_array_trim_30db"\
    --time_length 350\
    --train-data-fraction 0.8\
    --validation-data-fraction 0.1\
    --num-epochs 20\
    --batch-size 10\
    --num-hiddens 768\
    --num-residual-hiddens 32\
    --num-residual-layers 2\
    --embedding-dim 64\
    --num-embeddings 256\
    --speaker-embedding-dim 20\
    --commitment-cost 0.25\
    --decay 0\
    --learning-rate 1e-4
'''

############################################
## .. CHANGE THIS STUFF ONLY ..
############################################
code_base_dir  = '/home/ubuntu/speech2speech/'
training_exp_name = 'training_exp_original_dat_00' ## CHANGE THIS EVERY TIME

# All option choices for the script
choices = {}
choices['--data-dir'] = ['/home/ubuntu/speech2speech/data/raw/VCTK-Corpus']
choices['--spectrogram-dir'] = [\
    '/home/ubuntu/speech2speech/data/interim/spectogram_array_trim_30db',\
    '/home/ubuntu/speech2speech/data/interim/spectogram_array_path_trim_30db_ntft_512']

choices['--time-length'] = [50,100]
choices['--train-data-fraction'] = [0.8]
choices['--validation-data-fraction'] = [0.1]
choices['--num-epochs'] = [20]
choices['--batch-size'] = [10,40]
choices['--num-hiddens'] = [768]
choices['--num-residual-hiddens'] = [32]
choices['--num-residual-layers'] = [2]
choices['--embedding-dim'] = [64]
choices['--num-embeddings'] = [300]
choices['--speaker-embedding-dim'] = [20]
choices['--commitment-cost'] = [0.25]
choices['--decay'] = [0]
choices['--learning-rate'] = [1e-4]

choices_to_vary = ['--batch-size', '--time-length'] #change this everytime



############################################
## .. DONT CHANGE BELOW ..
############################################



############################################
## ------- Functions ---------
############################################
def get_all_combinations_from_two_lists(list1, list2):
    #print(' --- func begins ---')
    list1_permutations = itertools.permutations(list1, 1)
    all_combinations = []
    for each_permutation in list1_permutations:
        #print(each_permutation)
        #print(list2)
        zipped = zip(each_permutation, list2)
        #print(zipped)
        all_combinations.append(list(zipped))
    #print(' --- func ends ---')
    return all_combinations

def get_fixed_choice_value(fixed_choice, choices):
    opts = choices[fixed_choice]
    return opts[0]

def get_script_strings(choices, code_base):
    # list of all different script contents
    out_strs = []

    # Basic starting string for training scripts
    basic_str = '''\
#!/usr/bin/env bash

python {0}/scripts/train_model.py --verbose --debug\\
'''.format(code_base)

    all_choices = set(choices.keys())
    fixed_choices = all_choices - set(choices_to_vary)

    for fixed_choice in fixed_choices:
        basic_str += '    '  + fixed_choice + ' ' + '{}'.format(\
            get_fixed_choice_value(fixed_choice, choices)) + '\\\n'

    # Get combinations of all variable options
    all_option_string_combinations = []
    for variable_choice in choices_to_vary:
        opts = choices[variable_choice]
        option_string_combinations = get_all_combinations_from_two_lists(opts, list([variable_choice]))
        all_option_string_combinations.append( option_string_combinations )

    # Now, for each option for variable choices, write a completely
    # new script string
    for option_combos in itertools.product(*all_option_string_combinations):
        out_str = basic_str
        for option_combo in option_combos:
            out_str += '    '  + option_combo[0][1] + ' ' + '{}'.format(option_combo[0][0]) + '\\\n'

        # Append to main list storing different script strings
        out_strs.append(out_str)

    return out_strs


############################################
## ------ Setup : GLOBAL Variables -------
############################################

train_base_dir            = '{}/training/{}/'.format(code_base_dir, training_exp_name)
training_scripts_base_dir = '{}/scripts/{}/'.format(code_base_dir, training_exp_name)

# Make required directories
dirs_to_make = [train_base_dir, training_scripts_base_dir]

for d in dirs_to_make:
    try:
        os.makedirs(d)
    except: pass



############################################
## ------- Write out scripts -----
############################################
for idx, script_str in enumerate(get_script_strings(choices, code_base_dir)):

    print("Setting up test {}".format(idx))

    # Name of script and output dir for this training experiment
    out_dir = 'test_{:02d}'.format(idx)
    script_name = os.path.join(training_scripts_base_dir, out_dir) + '.sh'

    new_base_dir_path = os.path.join(train_base_dir, out_dir)
    # Create the new base dir
    try: os.makedirs(new_base_dir_path)
    except: pass

    # Write run script for this training experiment
    with open(script_name, 'w') as fout:
        fout.write(script_str)
        fout.write('    --base-dir {}\n'.format(new_base_dir_path))
    os.system('chmod +x {}'.format(script_name))
