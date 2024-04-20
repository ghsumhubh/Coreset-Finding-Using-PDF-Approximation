def binary_encoding(df):

    custom_value_mappings = {
    'red' :1, 'white': 0,
    }

    custom_column_mappings = {
        'type': 'is_red'
    }


    for column in df.columns:
        if set(df[column].unique()).issubset(set(custom_value_mappings.keys())):
            df[column] = df[column].map(custom_value_mappings)

            new_name = custom_column_mappings[column] if column in custom_column_mappings else column

            df = df.rename(columns={column: new_name})

    return df
