import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Some of the plots I use

def plot_sweep_heatmap(sweep_data, eta_vals, title,N,epochs):
    """Heatmap of final MSE scores."""
    
    df = pd.DataFrame(sweep_data, index=[f"{e:.1e}" for e in eta_vals]) # Rows: eta, Columns: Optimizers
    
    plt.figure(figsize=(14, 8), dpi=200)
    sns.heatmap(
        df, 
        annot=True, 
        fmt=".4f", 
        cmap="viridis_r", 
        cbar_kws={'label': 'Final MSE Score'}, 
        annot_kws={"size": 14}
    )
    plt.title(f'Hyperparameter Sweep: {title}, Sigmoid Activation,N={N}, Epochs={epochs}', fontsize=16)
    plt.ylabel('Learning Rate ($\eta$)')
    plt.xlabel('Optimizer')
    plt.show()


def plot_activation_sweep_heatmap(results_data, row_labels, title, eta_constant, N, epochs):

    df = pd.DataFrame(results_data).T 
    plt.figure(figsize=(10, 6)) 

    sns.heatmap(
        df, 
        annot=True, 
        fmt=".6f",
        cmap="viridis_r",
        cbar_kws={'label': 'Final MSE Score'}
    )

    plt.title(f'{title} (Fixed $\eta={eta_constant}$), N={N}, Epochs={epochs}') 
    plt.ylabel('Hidden Layer Activation Function')
    plt.xlabel('Optimizer')
    plt.show()
    plt.close()



