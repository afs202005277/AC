def create_lagged_features(dataframe, features_to_be_lagged, lag_years, group_by_column_name):
    tmp = "Teams" if group_by_column_name == 'tmID' else 'Players'
    print("Create Lagged Features " + tmp)
    # Create lagged features
    for feat in features_to_be_lagged:
        for year in range(1, lag_years + 1):
            dataframe[f'{feat}_Lag_{year}'] = dataframe.groupby(group_by_column_name)[feat].shift(year)

    # Fill NaN values in the newly created columns with 0
    lagged_features = [f'{feat}_Lag_{year}' for feat in features_to_be_lagged for year in
                       range(1, lag_years + 1)]
    if tmp == 'Teams':
        dataframe[lagged_features] = dataframe[lagged_features].fillna(-1)
    else:
        dataframe[lagged_features] = dataframe[lagged_features].fillna(0)

    if tmp == 'Teams':
        dataframe[[i for i in lagged_features if "playoff" in i]] = dataframe[
            [i for i in lagged_features if "playoff" in i]].replace(-1, 0.5)

    return dataframe