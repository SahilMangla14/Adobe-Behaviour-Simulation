import pandas as pd

# df = pd.read_csv('df_train.csv')
# df1 = pd.read_csv('ner_entity_300k_updated.csv')
# df2 = pd.read_csv('df_company_rank.csv')
# df3 = pd.read_csv('df_entity_rank.csv')
# df4 = pd.read_csv('pca.csv')
# df4 = pd.read_csv('df_train.csv')
# df5 = pd.read_csv('df_train2.csv')
# print(len(df1))
# print(len(df2))
# print(len(df3))
# print(len(df4))
# print(df4.columns)
# print(df5.columns)
# print(df4)
# print(df5)

# features_to_scale = ['epoch_time', 'total_rank_entity', 'company_rank']
# for index, feature in enumerate(features_to_scale):
#     print(feature)
    
    
# df1 = pd.read_csv('./problem_1_test_dataset/problem_1_test_dataset/behaviour_simulation_test_time.csv')
# df2 = pd.read_csv('./ner_entity_300k_updated.csv')

# all_users_in_df2 = df1['username'].isin(df2['username']).all()

# # Count of unique names in df1 and df2
# unique_names_count_df1 = df1['username'].nunique()
# unique_names_count_df2 = df2['username'].nunique()


# # Find usernames in df1 that are not in df2
# usernames_not_in_df2 = df1[~df1['username'].isin(df2['username'])]

# # Count of unique usernames not in df2
# count_usernames_not_in_df2 = usernames_not_in_df2['username'].nunique()

# print("Count of usernames in df1 but not in df2:", count_usernames_not_in_df2)

# print("All users from df1 present in df2:", all_users_in_df2)
# print("Count of unique names in df1:", unique_names_count_df1)
# print("Count of unique names in df2:", unique_names_count_df2)

# # Find usernames common to both df1 and df2
# common_usernames = df1[df1['username'].isin(df2['username'])]['username'].unique()

# # Print the common usernames
# print("Usernames in both df1 and df2:")
# print(common_usernames.tolist())
# print(len(common_usernames.tolist()))

# print("\n\n\n")

# all_users_in_df2 = df1['inferred company'].isin(df2['inferred company']).all()

# # Count of unique names in df1 and df2
# unique_names_count_df1 = df1['inferred company'].nunique()
# unique_names_count_df2 = df2['inferred company'].nunique()


# # Find usernames in df1 that are not in df2
# usernames_not_in_df2 = df1[~df1['inferred company'].isin(df2['inferred company'])]

# # Count of unique usernames not in df2
# count_usernames_not_in_df2 = usernames_not_in_df2['inferred company'].nunique()

# print("Count of company in df1 but not in df2:", count_usernames_not_in_df2)

# print("All companies from df1 present in df2:", all_users_in_df2)
# print("Count of unique company in df1:", unique_names_count_df1)
# print("Count of unique company in df2:", unique_names_count_df2)

# # Find usernames common to both df1 and df2
# common_usernames = df1[df1['inferred company'].isin(df2['inferred company'])]['inferred company'].unique()

# # Print the common usernames
# print("company in both df1 and df2:")
# print(common_usernames.tolist())
# print(len(common_usernames.tolist()))

# # common_companies = pd.merge(df1, df2, on='inferred company', how='inner')['inferred company']
# # print("Companies present in both df1 and df2:")
# # print(common_companies)


# # company_to_check = 'bbc'

# # if company_to_check in df2['inferred company'].values:
# #     print(f"{company_to_check} is present in df2.")
# # else:
# #     print(f"{company_to_check} is not present in df2.")


# df = pd.read_csv('./data_train/df_train.csv')
# columns_to_drop = [col for col in df.columns if col.startswith('username')]

# # Drop the selected columns
# df.drop(columns=columns_to_drop, inplace=True)
# df = df.drop(columns = ['epoch_time','total_rank_entity','company_rank','likes'])

# print(len(df))
# print(len(df.columns))


df1 = pd.read_csv('./data_company/behaviour_simulation_test_company.csv')
df2 = pd.read_csv('./Predictions/company_dataset_predictions.csv')

df = pd.concat([df1,df2], axis = 1)

df.to_csv('./data_company/company_pred_likes.csv')

df1 = pd.read_csv('./data_time/behaviour_simulation_test_time.csv')
df2 = pd.read_csv('./Predictions/time_dataset_predictions.csv')

df = pd.concat([df1,df2], axis = 1)

df.to_csv('./data_time/time_pred_likes.csv')

