import pandas as pd







def main(data_path=r'../models/unsupervised_states') :
    cytof_cluster_data = pd.read_excel(r'../models/unsupervised_states/samples_data/CyTOF.features.and.clusters.info.xlsx')
    read_cytof_data = pd.read_excel(r'../models/unsupervised_states/samples_data/filtered.esetALL.CyTOF.abundance.only.xlsx')

    cytof_cluster_data = cytof_cluster_data[cytof_cluster_data["dataSource"] != "GRAN"]

    cytof_cluster_no_antigens = _get_numeric_feature_id(cytof_cluster_data)
    cells_biggest_clusters_num_list = cytof_cluster_no_antigens.groupby("cellType")["featureID"].max().tolist()
    cells_biggest_clusters_idx_list = [idx for idx, cluster in cytof_cluster_no_antigens["featureID"].items() if cluster in cells_biggest_clusters_num_list]

    relevant_idx_to_cell_type_dict = {idx:cell_type for idx,cell_type in cytof_cluster_no_antigens.ix[cells_biggest_clusters_idx_list]["cellType"].items()}
    print('pass')


def _get_numeric_feature_id(cytof_cluster_data):
    cytof_cluster_no_antigens = cytof_cluster_data[cytof_cluster_data.index.str.len() < 11]
    cytof_cluster_no_antigens["featureID"] = cytof_cluster_no_antigens["featureID"].str.slice(3)
    cytof_cluster_no_antigens = cytof_cluster_no_antigens[
        cytof_cluster_no_antigens.featureID.apply(lambda x: x.isnumeric())]
    cytof_cluster_no_antigens["featureID"] = cytof_cluster_no_antigens["featureID"].apply(
        (lambda x: int(x) if x.isdigit() else x))

    return cytof_cluster_no_antigens






if __name__ == '__main__':
    main()