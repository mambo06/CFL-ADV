dirs = [
             'adult',
             'aloi',
             # 'california_housing',
             'covtype',
             'epsilon',
             'helena',
             'higgs_small',
             'jannis',
             # 'microsoft',
             # 'yahoo',
             # 'year'
            ]
for item in dirs:
    for pc in ['0.2','0.5']:
        for pl in ['0.1','0.5','2']:
            for rl in ['0.8','0.8']:
                print("python -W ignore train.py -e 50 -c 8 -d", item,'-pc',pc,'-pl',pl,'-rl',rl)