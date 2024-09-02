#Import library yang diperlukan
import streamlit as st
import pandas as pd
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

import fpcommon as fpc



#Perhitungan Association Rule
def association_rules(df, metric="confidence", min_threshold=0.8, support_only=False, decimals=4):
    if not df.shape[0]:
        raise ValueError("The DataFrame is empty.")

    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError("DataFrame must contain 'support' and 'itemsets' columns.")

    def conviction_helper(sAC, sA, sC):
        confidence = sAC / sA
        conviction = np.empty(confidence.shape, dtype=float)
        if not len(conviction.shape):
            conviction = conviction[np.newaxis]
            confidence = confidence[np.newaxis]
            sAC = sAC[np.newaxis]
            sA = sA[np.newaxis]
            sC = sC[np.newaxis]
        conviction[:] = np.inf
        conviction[confidence < 1.0] = (1.0 - sC[confidence < 1.0]) / (1.0 - confidence[confidence < 1.0])
        return conviction

    def zhangs_metric_helper(sAC, sA, sC):
        denominator = np.maximum(sAC * (1 - sA), sA * (sC - sAC))
        numerator = metric_dict["leverage"](sAC, sA, sC)
        with np.errstate(divide="ignore", invalid="ignore"):
            zhangs_metric = np.where(denominator == 0, 0, numerator / denominator)
        return zhangs_metric

    metric_dict = {
        "support": lambda sAC, _, __: sAC,
        "confidence": lambda sAC, sA, _: sAC / sA,
        "lift": lambda sAC, sA, sC: metric_dict["confidence"](sAC, sA, sC) / sC,
    }

    columns_ordered = [
        "support",
        "confidence",
        "lift",
    ]

    if support_only:
        metric = "support"
    else:
        if metric not in metric_dict.keys():
            raise ValueError(f"Metric must be 'confidence' or 'lift', got '{metric}'")

    keys = df["itemsets"].values
    values = df["support"].values
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    for k in frequent_items_dict.keys():
        sAC = frequent_items_dict[k]
        for idx in range(len(k) - 1, 0, -1):
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)

                if support_only:
                    sA = None
                    sC = None
                else:
                    try:
                        sA = frequent_items_dict[antecedent]
                        sC = frequent_items_dict[consequent]
                    except KeyError as e:
                        s = (str(e) + " You are likely getting this error")
                        raise KeyError(s)

                score = metric_dict[metric](sAC, sA, sC)
                if score >= min_threshold:
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append([sAC, sA, sC])

    if not rule_supports:
        return pd.DataFrame(columns=["antecedents", "consequents"] + columns_ordered)

    else:
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedents, rule_consequents)),
            columns=["antecedents", "consequents"],
        )

        if support_only:
            sAC = rule_supports[0]
            for m in columns_ordered:
                df_res[m] = np.nan
            df_res["support"] = sAC

        else:
            sAC = rule_supports[0]
            sA = rule_supports[1]
            sC = rule_supports[2]
            for m in columns_ordered:
                df_res[m] = np.round(metric_dict[m](sAC, sA, sC), decimals)

            # df_res["precision"] = df_res.apply(
            #     lambda row: row["support"] / (len(row["consequents"]) + len(row["antecedents"])), axis=1
            # )
            # df_res["recall"] = df_res.apply(
            #     lambda row: row["support"] / len(row["antecedents"]), axis=1
            # )

        df_res["antecedents"] = df_res["antecedents"].apply(lambda x: list(x) if isinstance(x, frozenset) else x)
        df_res["consequents"] = df_res["consequents"].apply(lambda x: list(x) if isinstance(x, frozenset) else x)

        return df_res
 
 #Menghasilkan Kombinasi Baru
def generate_new_combinations(old_combinations):
    
    items_types_in_previous_step = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        max_combination = old_combination[-1]
        mask = items_types_in_previous_step > max_combination
        valid_items = items_types_in_previous_step[mask]
        old_tuple = tuple(old_combination)
        for item in valid_items:
            yield from old_tuple
            yield item


def generate_new_combinations_low_memory(old_combinations, X, min_support, is_sparse):
   
    items_types_in_previous_step = np.unique(old_combinations.flatten())
    rows_count = X.shape[0]
    threshold = min_support * rows_count
    for old_combination in old_combinations:
        max_combination = old_combination[-1]
        mask = items_types_in_previous_step > max_combination
        valid_items = items_types_in_previous_step[mask]
        old_tuple = tuple(old_combination)
        if is_sparse:
            mask_rows = X[:, old_tuple].toarray().all(axis=1)
            X_cols = X[:, valid_items].toarray()
            supports = X_cols[mask_rows].sum(axis=0)
        else:
            mask_rows = X[:, old_tuple].all(axis=1)
            supports = X[mask_rows][:, valid_items].sum(axis=0)
        valid_indices = (supports >= threshold).nonzero()[0]
        for index in valid_indices:
            yield supports[index]
            yield from old_tuple
            yield valid_items[index]

#Proses Perhitungan Apriori
def apriori(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0, low_memory=False):
    # Fungsi untuk menghitung support
    def _support(data, total_rows, is_sparse):
        return np.sum(data, axis=0) / total_rows

    # Validasi nilai min_support
    if min_support <= 0.0:
        raise ValueError("`min_support` harus bernilai positif dalam rentang (0, 1].")

    fpc.valid_input_check(df)

    # Menangani data yang bersifat sparse atau tidak
    if hasattr(df, "sparse"):
        X = df.sparse.to_coo().tocsc() if df.size != 0 else df.values
        is_sparse = True
    else:
        X = df.values
        is_sparse = False

    # Menghitung support untuk 1-itemset
    support = _support(X, X.shape[0], is_sparse)
    itemset_indices = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: itemset_indices[support >= min_support].reshape(-1, 1)}
    max_itemset = 1
    total_rows = float(X.shape[0])
    all_ones = np.ones((int(total_rows), 1))

    # Loop untuk menemukan itemset yang sering muncul (frequent itemsets)
    while max_itemset and max_itemset < (max_len or float("inf")):
        next_max_itemset = max_itemset + 1

        if low_memory:
            combin = generate_new_combinations_low_memory(
                itemset_dict[max_itemset], X, min_support, is_sparse)
            combin = np.fromiter(combin, dtype=int).reshape(-1, next_max_itemset + 1)
            if combin.size == 0:
                break
            if verbose:
                print(f"\rProcessing {combin.size} combinations | Sampling itemset size {next_max_itemset}", end="")
            itemset_dict[next_max_itemset] = combin[:, 1:]
            support_dict[next_max_itemset] = combin[:, 0].astype(float) / total_rows
            max_itemset = next_max_itemset
        else:
            combin = generate_new_combinations(itemset_dict[max_itemset])
            combin = np.fromiter(combin, dtype=int).reshape(-1, next_max_itemset)
            if combin.size == 0:
                break
            if verbose:
                print(f"\rProcessing {combin.size} combinations | Sampling itemset size {next_max_itemset}", end="")

            if is_sparse:
                bools = X[:, combin[:, 0]] == all_ones
                for n in range(1, combin.shape[1]):
                    bools &= (X[:, combin[:, n]] == all_ones)
            else:
                bools = np.all(X[:, combin], axis=2)

            support = _support(np.array(bools), total_rows, is_sparse)
            mask = (support >= min_support).reshape(-1)
            if any(mask):
                itemset_dict[next_max_itemset] = np.array(combin[mask])
                support_dict[next_max_itemset] = np.array(support[mask])
                max_itemset = next_max_itemset
            else:
                break

    # Menggabungkan dan mengembalikan hasil
    all_results = []
    for k in sorted(itemset_dict):
        support_series = pd.Series(support_dict[k])
        itemsets_series = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")
        result = pd.concat((support_series, itemsets_series), axis=1)
        all_results.append(result)

    result_df = pd.concat(all_results)
    result_df.columns = ["support", "itemsets"]
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        result_df["itemsets"] = result_df["itemsets"].apply(
            lambda x: frozenset([mapping[i] for i in x])
        )
    result_df = result_df.reset_index(drop=True)

    if verbose:
        print()  # menambah baris baru jika verbose digunakan

    return result_df


def encode(x):
    """Encodes values as 0 (if <= 0) or 1 (if >= 1)."""
    if x <= 0:
        return 0
    elif x >= 1:
        return 1

def preprocess_data(df):

    filtered = df.copy()
    # value_counts_result = data['ID CUSTOMER'].value_counts()
    # table = value_counts_result.rename_axis('ID CUSTOMER').reset_index(name='Count')
    # filtered = data[data['ID CUSTOMER'].isin(value_counts_result[value_counts_result <= 3 ].index)]
    # data['Date'] = pd.to_datetime(data['Date'])
    #filter Data Ke Bulan
    # Memfilter data untuk transaksi selama 6 bulan pertama tahun 2021
    # start_date = '2021-01-01' 
    # end_date = '2021-03-30'
    # Filter DataFrame berdasarkan rentang tanggal
    #filtered = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    
    #filtered_data = filtered
    

    filtered = filtered.loc[:, ['Date','ID CUSTOMER', 'NAMA BARANG', 'HARGA SATUAN', 'SUBTOTAL']]
    filtered = filtered.drop_duplicates(keep=False)
    filtered['Qty'] = filtered['SUBTOTAL'] / filtered['HARGA SATUAN']
    # filtered = filtered.loc[:, ['ID CUSTOMER', 'NAMA BARANG', 'Qty']]
    print(f"Encoded data shape Filtered Data: {filtered.shape}") 
    filtered = filtered.rename(columns={"ID CUSTOMER": "idTransaksi", "NAMA BARANG": "namaBarang"})

    pivot_table_filtered = filtered.pivot_table(
        index='idTransaksi', columns='namaBarang', values='Qty', aggfunc='sum'
    ).fillna(0)

    pivot_table_filtered = pivot_table_filtered.reset_index()
    pivot_table_filtered = pivot_table_filtered.drop(pivot_table_filtered.columns[0], axis=1)
    pivot_table_filtered = pivot_table_filtered.astype("int64")
    pivot_table_filtered = pivot_table_filtered.applymap(encode)

    return pivot_table_filtered

def run_apriori(data, min_support, min_confidence):
  
  
  print(f"Encoded data shape: {data.shape}") 

  frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
  frequent_itemsets.sort_values("support", ascending=False)
  rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
  rules.sort_values(['confidence'], ascending = False, inplace =True)

  rules.reset_index(drop=True, inplace=True)
  rules.rename(columns={"antecedents": "condition", "consequents": "result"},inplace=True)
  
  return frequent_itemsets, rules

# def calculate_precision_recall(rules):
#     precision_list = []
#     recall_list = []
#     for _, row in rules.iterrows():
#         antecedents = row['condition']
#         consequents = row['result']
#         total_transactions = len(data)
#         tp = sum(data[consequents].any(axis=1) & data[antecedents].any(axis=1))
#         fp = sum(data[consequents].any(axis=1) & ~data[antecedents].any(axis=1))
#         fn = sum(~data[consequents].any(axis=1) & data[antecedents].any(axis=1))
#         precision = tp / (tp + fp) if (tp + fp) != 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) != 0 else 0
#         precision_list.append(precision)
#         recall_list.append(recall)
#     rules['precision'] = precision_list
#     rules['recall'] = recall_list
#     return rules

#st.header("Market Basket Analysis User Interface")
st.markdown("""<style> .big-title { font-size: 40px !important; } </style>
<h1 class="big-title">Market Basket Analysis <br> User Interface</h1>""", unsafe_allow_html=True)

# Upload data
uploaded_file = st.file_uploader("Unggah data anda (xlsx format)", type="xlsx")
# start_date = st.date_input("When's your birthday")
# end_date = st.date_input("Wh")
# User input for minimum support and confidence
#min_support_slider = st.number_input("Minimum Support",min_value= 0.001,  placeholder="Masukkan Minimum Support")
st.markdown("**Minimum Support** :gray[(mengukur seberapa sering suatu itemset (kumpulan item) muncul)]")
min_support_str = st.text_input("rekomendasi nilai : 0.001 - 0.06",placeholder="Masukkan Minimum Support")

try:
    min_support = float(min_support_str)
except ValueError:
    min_support = 0.001

st.markdown("**Minimum Confidence**\n:gray[(kekuatan hubungan antar item dalam itemset)]")

min_confidence_slider = st.number_input("Rekomendasi nilai : 0.01 - 1",value=None,placeholder="Masukkan Minimum Confidence")
start_date = st.date_input('Tanggal/bulan/tahun-Mulai')
end_date = st.date_input('Tanggal/bulan/tahun-Akhir')

# Pilih Rentang Tanggal



if st.button("Jalankan"):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)  # Assuming data is in a pandas DataFrame
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Preprocess the data
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        processed_data = preprocess_data(data.copy())  # Avoid modifying original data

        frequent_itemsets, rules = run_apriori(processed_data.copy(), min_support, min_confidence_slider)

        # Display the preprocessed data
        # st.dataframe(processed_data.copy())
        # st.subheader("Frequent Itemsets:")
        # st.dataframe(frequent_itemsets)

        
        st.subheader("Tabel Asosiasi:")
        st.dataframe(rules)
        st.write('--------------------------------------------------------------------------------')
        if not rules.empty:
            for _, row in rules.iterrows():
                antecedents = ', '.join(list(row['condition']))
                consequents = ', '.join(list(row['result']))
                st.write(f"Jika Customer Membeli {antecedents}, Maka juga akan membeli {consequents}, dengan nilai confidence sebesar {row['confidence']:.2f}")
        else:
            st.write("No association rules found with the given parameters.")
        #st.dataframe(rules)

        # st.subheader("Value Count:")
        # st.dataframe(value)

        # st.subheader("Data Filtered:")
        # st.dataframe(rules)
    else:
        st.warning("Silakan unggah file XLSX yang berisi data Anda!")

        # You can now use processed_data for Apriori analysis (uncomment these lines)
        # min_support = st.slider('Minimum Support', min_val=0.01, max_val=1.0, step=0.01)
        # frequent_itemsets = apriori(processed_data, min_support=min_support, use_colnames=True)
        # association_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
        # st.dataframe(association_rules)

# if st.button("Jalankan"):
#     if uploaded_file is not None:
#         data = pd.read_excel(uploaded_file)
#         processed_data = preprocess_data(data.copy())
#         frequent_itemsets, rules = run_apriori(processed_data, min_support, min_confidence_slider)
#         rules = calculate_precision_recall(rules)
        
#         st.markdown("### Frequent Itemsets")
#         st.dataframe(frequent_itemsets)
        
#         st.markdown("### Association Rules")
#         st.dataframe(rules)
        
#         st.markdown("### Precision and Recall")
#         st.dataframe(rules[['condition', 'result', 'precision', 'recall']])