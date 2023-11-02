def json_to_df(data, output_path):
    '''
    function that takes a json
    and saves it as a df in output path
    also returns the final_df 
    '''
    headers = [k for k,v in data.items()]

    header_data = [data[header] for header in headers]

    all_dfs = []
    for i in range(len(headers)):
        df = pd.DataFrame(columns=[headers[i]])
        df[headers[i]] = header_data[i]
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, axis=1)
    final_df.to_excel(f'{output_path}.xlsx',index=False)

    return final_df
