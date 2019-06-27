import pandas as pd

log = open("check_nn_results", "w")
log.close()

real_data = pd.read_csv('results.csv')
nn_output = pd.read_csv('results_nn.csv')

nn_output.rename(columns={'Unnamed: 0':'case number'}, inplace=True)
print(real_data)
print(nn_output)

test_cases = real_data.iloc[nn_output['case number'],:].astype(float).reset_index()

#get only Cd and Cl data from both dataframes
compare = test_cases.loc[:,['Cd','Cl']].join(nn_output.loc[:,['unscaled_Cd','unscaled_Cl']].rename(columns={'unscaled_Cd':'pred_Cd', 'unscaled_Cl':'pred_Cl'}))

#test_cases['pred_Cd','pred_Cl'] = nn_output.loc[:,['real_Cd','real_Cl']].astype(float)
#test_cases['pred_Cl'] = nn_output.loc[:,'real_Cl'].astype(float)


compare['diff_Cd'] = compare['pred_Cd'] - compare['Cd']
compare['diff_Cl'] = compare['pred_Cl'] - compare['Cl']

compare['error_Cd (%)'] = (compare['diff_Cd'] / compare['Cd']) * 100
compare['error_Cl (%)'] = (compare['diff_Cl'] / compare['Cl']) * 100


print(compare)
