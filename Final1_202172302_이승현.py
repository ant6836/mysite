#%%
import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from hmmlearn import hmm
from hmm_classifier import HMM_classifier

#%%
data = pd.read_csv("./kddcup.data.corrected", header=None)

#%%
data.columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "intrusion_type",
]

#%%
intrusion_type_code = {
    "ftp_write.": "r2l",
    "guess_passwd.": "r2l",
    "imap.": "r2l",
    "multihop.": "r2l",
    "phf.": "r2l",
    "spy.": "r2l",
    "warezclient.": "r2l",
    "warezmaster.": "r2l",
    "back.": "dos",
    "land.": "dos",
    "neptune.": "dos",
    "pod.": "dos",
    "smurf.": "dos",
    "teardrop.": "dos",
    "buffer_overflow.": "u2r",
    "loadmodule.": "u2r",
    "perl.": "u2r",
    "rootkit.": "u2r",
    "normal.": "normal",
    "ipsweep.": "probe",
    "nmap.": "probe",
    "portsweep.": "probe",
    "satan.": "probe",
}

value_dict = {"dos": 1, "normal": 0}

#%%
data["intrusion_code"] = data["intrusion_type"].map(intrusion_type_code)
data["value"] = data["intrusion_code"].map(value_dict)

#%%
data_dos = data[data["value"] == 1]
data_dos.reset_index(drop=True, inplace=True)

#%%
data_normal = data[data["value"] == 0]
data_normal.reset_index(drop=True, inplace=True)

#%%
data_sampled_dos = data_dos.sample(n=100000, ignore_index=True)
data_sampled_normal = data_normal.sample(n=100000, ignore_index=True)

#%%
use_column_list = [
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

df_dos = data_sampled_dos[use_column_list]
df_normal = data_sampled_normal[use_column_list]

#%%
for col in use_column_list:
    df_dos[col] = df_dos[col] + K.epsilon()
    df_normal[col] = df_normal[col] + K.epsilon()

#%%
log_dos = df_dos.copy()
log_normal = df_normal.copy()

#%%
for col in use_column_list:
    log_dos[col] = np.log(log_dos[col])
    log_normal[col] = np.log(log_normal[col])

# %%
pca = PCA(n_components=5)

pca_dos = pca.fit_transform(log_dos)
pca_normal = pca.fit_transform(log_normal)

#%%
dos_label_dict, normal_label_dict = dict(), dict()

kmeans_dos = KMeans(n_clusters=60, random_state=0).fit(pca_dos)
kmeans_normal = KMeans(n_clusters=60, random_state=0).fit(pca_normal)

#%%
for i, label in enumerate(kmeans_dos.labels_):
    if label not in dos_label_dict.keys():
        dos_label_dict[label] = list()
    dos_label_dict[label].append(pca_dos[i].tolist())

for i, label in enumerate(kmeans_normal.labels_):
    if label not in normal_label_dict.keys():
        normal_label_dict[label] = list()
    normal_label_dict[label].append(pca_normal[i].tolist())

#%%
sorted_dos_label_dict = sorted(dos_label_dict.items())
sorted_normal_label_dict = sorted(normal_label_dict.items())

#%%
train = list()
for vec in sorted_dos_label_dict[:50]:
    train.append(vec[1])
for vec in sorted_normal_label_dict[:50]:
    train.append(vec[1])

#%%
test = list()
for vec in sorted_dos_label_dict[50:]:
    test.append(vec[1])
for vec in sorted_normal_label_dict[50:]:
    test.append(vec[1])

#%%
model = HMM_classifier(hmm.GaussianHMM())
model.fit(train, [0 for _ in range(50)] + [1 for _ in range(50)])

y_pred = []
for vec in test:
    y_pred.append(model.predict(vec))
