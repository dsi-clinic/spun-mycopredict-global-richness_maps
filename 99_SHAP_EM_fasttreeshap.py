import pandas as pd
import numpy as np
import shap        
from shap.plots import colors
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import multiprocessing
from contextlib import contextmanager
import fasttreeshap
import datetime

today = datetime.date.today().strftime("%Y%m%d")

# Constants
classProperty = 'ectomycorrhizal_richness'
df = pd.read_csv('data/20260123_ectomycorrhizal_only_alphaearth_center.csv')

# Variables to include in the model
envCovariateList = [
'A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09',
'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19',
'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29',
'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39',
'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49',
'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59',
'A60', 'A61', 'A62', 'A63'
]

# Rename variables in covariateList to increase readability
envCovariateListRenamed = [
    'AlphaEarth A00', 'AlphaEarth A01', 'AlphaEarth A02', 'AlphaEarth A03', 'AlphaEarth A04',
    'AlphaEarth A05', 'AlphaEarth A06', 'AlphaEarth A07', 'AlphaEarth A08', 'AlphaEarth A09',
    'AlphaEarth A10', 'AlphaEarth A11', 'AlphaEarth A12', 'AlphaEarth A13', 'AlphaEarth A14',
    'AlphaEarth A15', 'AlphaEarth A16', 'AlphaEarth A17', 'AlphaEarth A18', 'AlphaEarth A19',
    'AlphaEarth A20', 'AlphaEarth A21', 'AlphaEarth A22', 'AlphaEarth A23', 'AlphaEarth A24',
    'AlphaEarth A25', 'AlphaEarth A26', 'AlphaEarth A27', 'AlphaEarth A28', 'AlphaEarth A29',
    'AlphaEarth A30', 'AlphaEarth A31', 'AlphaEarth A32', 'AlphaEarth A33', 'AlphaEarth A34',
    'AlphaEarth A35', 'AlphaEarth A36', 'AlphaEarth A37', 'AlphaEarth A38', 'AlphaEarth A39',
    'AlphaEarth A40', 'AlphaEarth A41', 'AlphaEarth A42', 'AlphaEarth A43', 'AlphaEarth A44',
    'AlphaEarth A45', 'AlphaEarth A46', 'AlphaEarth A47', 'AlphaEarth A48', 'AlphaEarth A49',
    'AlphaEarth A50', 'AlphaEarth A51', 'AlphaEarth A52', 'AlphaEarth A53', 'AlphaEarth A54',
    'AlphaEarth A55', 'AlphaEarth A56', 'AlphaEarth A57', 'AlphaEarth A58', 'AlphaEarth A59',
    'AlphaEarth A60', 'AlphaEarth A61', 'AlphaEarth A62', 'AlphaEarth A63'
]

project_vars = [
'sequencing_platform454Roche',
'sequencing_platformIllumina',
'sequencing_platformIonTorrent',
'sequencing_platformPacBio',
'sample_typerhizosphere_soil',
'sample_typesoil',
'sample_typetopsoil',
'primers5_8S_Fun_ITS4_Fun',
'primersfITS7_ITS4',
'primersfITS9_ITS4',
'primersgITS7_ITS4',
'primersgITS7_ITS4_then_ITS9_ITS4',
'primersgITS7_ITS4_ITS4arch',
'primersgITS7_ITS4m',
'primersgITS7_ITS4ngs',
'primersgITS7ngs_ITS4ngsUni',
'primersITS_S2F___ITS3_mixed_1_1_ITS4',
'primersITS1_ITS4',
'primersITS1F_ITS4',
'primersITS1F_ITS4_then_fITS7_ITS4',
'primersITS1F_ITS4_then_ITS3_ITS4',
'primersITS1ngs_ITS4ngs_or_ITS1Fngs_ITS4ngs',
'primersITS3_KYO2_ITS4',
'primersITS3_ITS4',
'primersITS3ngs1_to_5___ITS3ngs10_ITS4ngs',
'primersITS3ngs1_to_ITS3ngs11_ITS4ngs',
'primersITS86F_ITS4',
'primersITS9MUNngs_ITS4ngsUni',
'area_sampled',
'extraction_dna_mass',
]

# Rename variables in df to increase readability
column_names = dict(zip(envCovariateList, envCovariateListRenamed))
df = df.rename(columns=column_names)

# Create final list of covariates
covariateList = envCovariateListRenamed + project_vars

# Subset columns from df
df = df[covariateList + [classProperty]]

# Set categorical variables
for cat in project_vars:
    df[cat] = df[cat].astype('category')

# Load data and labels
X = df[covariateList]
y = df[classProperty]

# Train Random Forest models and calculate SHAP values
def calculate_shap_values(rep):
    grid_search_results = pd.read_csv('output/20260122_ectomycorrhizal_richness_grid_search_results.csv')
    VPS = int(grid_search_results['cName'][rep].split('VPS')[1].split('_')[0])
    LP = int(grid_search_results['cName'][rep].split('LP')[1].split('_')[0])

    hyperparameters = {
        'n_estimators': 250,
        'min_samples_split': LP,
        'max_features': VPS,
        'max_samples': 0.632,
        'random_state': 42
    }

    classifier = RandomForestRegressor()
    classifier.set_params(**hyperparameters)

    classifier.fit(X, y)

    explainer = fasttreeshap.TreeExplainer(classifier)

    shap_values = explainer(df[covariateList])

    return shap_values.values

@contextmanager
def poolcontext(*args, **kwargs):
		"""This just makes the multiprocessing easier with a generator."""
		pool = multiprocessing.Pool(*args, **kwargs)
		yield pool
		pool.terminate()

NPROC = 10

if __name__ == '__main__':
    reps = list(range(0, 10))
    with poolcontext(NPROC) as pool:
        try:
            with np.load('shap_values_ECM_richness.npz') as data:
                shap_values_list = [data[f'arr_{i}'] for i in range(len(data.keys()))]
        except Exception as e:
            shap_values_list = pool.map(calculate_shap_values, reps)
              
            # Save SHAP values to file
            np.savez('shap_values_ECM_richness.npz', *shap_values_list)

    # Plot 1: SHAP summary plot, with all features
    plt.figure()
    shap.summary_plot(np.mean(shap_values_list, axis=0), pd.DataFrame(data=df, columns=covariateList), show = False, sort = True)
    plt.xlabel('Mean absolute SHAP value')
    plt.tight_layout()
    # plt.show()
    plt.savefig('figures/shap/'+today+'_'+'ectomycorrhizal_richness_shap_summary_plots_full.png', dpi=300)

    # Plot 2: SHAP summary plot, with project_vars removed
    # Calculate mean SHAP values
    mean_shap_values = np.mean(shap_values_list, axis=0)
    # Get the indices of the features to drop
    drop_indices = [i for i, feat in enumerate(covariateList) if feat in project_vars]

    # Create a mask where only the features not in project_vars are True
    mask = np.ones(len(covariateList), dtype=bool)
    mask[drop_indices] = False

    # Create a new dataframe without the features to drop
    df_filtered = df[envCovariateListRenamed]

    # Filter the mean SHAP values
    mean_shap_values_filtered = mean_shap_values[:, mask]
    
    # Plot and save figure to file
    plt.figure()
    shap.summary_plot(mean_shap_values_filtered, df_filtered, show=False, sort=True)
    plt.xlabel('Mean absolute SHAP value')
    plt.tight_layout()
    # plt.show()
    plt.savefig('figures/shap/'+today+'_'+'ectomycorrhizal_richness_shap_summary_plots_projectRemoved.png', dpi=300)

    # Plot 3: SHAP summary plot, with project_vars grouped together
    # Sum 'project_vars' SHAP values together
    project_shap_values = np.sum(mean_shap_values[:, len(covariateList) - len(project_vars):], axis=1).reshape(-1, 1)

    # Get SHAP values for other features
    other_shap_values = mean_shap_values[:, :len(covariateList) - len(project_vars)]

    # Combine 'project_vars' SHAP values with other features
    combined_shap_values = np.hstack([other_shap_values, project_shap_values])

    # Create new feature names list
    new_feature_names = envCovariateListRenamed + ["project_vars"]

    # Create a df where project vars are Nan
    df_project_vars_grouped = df[envCovariateListRenamed]
    df_project_vars_grouped.loc[:, 'Project Variables'] = np.NaN

    plt.figure()
    shap.summary_plot(combined_shap_values, features = df_project_vars_grouped, sort=True, show = False)
    plt.xlabel('Mean absolute SHAP value')
    plt.tight_layout()
    plt.savefig('figures/shap/'+today+'_'+'ectomycorrhizal_richness_shap_summary_plots_projectGrouped.png', dpi=300)

    # # Plot 4: SHAP dependence plots for the top 6 features
    # # Create SHAP explanation object        
    # explanation = shap.Explanation(values=mean_shap_values_filtered,
    #             # base_values=shap_values_list[0].base_values,
    #             data=pd.DataFrame(data=df[envCovariateListRenamed + [classProperty]], columns=envCovariateListRenamed + [classProperty]),
    #             feature_names=list(df[envCovariateListRenamed + [classProperty]].columns))

    # # Get the top 6 most important features
    # importance = np.abs(explanation.values).mean(0)
    # top_6 = np.argsort(-importance)[:6]

    # # Create a multipanelled figure of the top 6 features
    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    # # Plot
    # for i, feature_idx in enumerate(top_6):
    #     shap.dependence_plot(envCovariateListRenamed[feature_idx], explanation.values, X[envCovariateListRenamed], ax=axes[i // 3, i % 3], interaction_index = 'auto', show=False)
    #     plt.tight_layout()

    # # Save figure to file
    # plt.savefig('figures/20240620_ectomycorrhizal_rwr_shap_scatter_plots_wInteraction.png', dpi=300)

    # # Plot 5: SHAP dependence plots for the top 6 features, without interaction
    # # Create a multipanelled figure of the top 6 features
    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    # # Plots without interaction
    # for i, feature_idx in enumerate(top_6):
    #     shap.dependence_plot(envCovariateListRenamed[feature_idx], explanation.values, X[envCovariateListRenamed], ax=axes[i // 3, i % 3], interaction_index = None, show=False)
    #     plt.tight_layout()

    # # Save figure to file
    # plt.savefig('figures/20240620_ectomycorrhizal_rwr_shap_scatter_plots.png', dpi=300)

    # # Plot 6: SHAP bar plot for the top 12 features, with project_vars grouped together
    # plt.figure()
    # shap.summary_plot(combined_shap_values, features = df_project_vars_grouped, plot_type = 'bar', sort=True, show = False, max_display=12)
    # plt.xlabel('Mean absolute SHAP value')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('figures/20240118_ectomycorrhizal_richness_shap_bar_plots_projectGrouped.png', dpi=300)

    mean_shap_values = np.mean(np.abs(combined_shap_values), axis=0)

    # Create a DataFrame with feature names from df_project_vars_grouped and their corresponding mean SHAP values
    df_mean_shap_values = pd.DataFrame({
        'Feature': df_project_vars_grouped.columns,
        'Mean SHAP Value': mean_shap_values
    })

    # Sort by absolute mean SHAP value
    df_mean_shap_values = df_mean_shap_values.reindex(df_mean_shap_values['Mean SHAP Value'].sort_values(ascending=False).index)

    # Write to file
    df_mean_shap_values.to_csv('figures/shap/'+today+'_'+'ectomycorrhizal_richness_mean_shap_values.csv', index=False)
