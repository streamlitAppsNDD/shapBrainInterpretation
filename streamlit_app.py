import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

sys.stdout = open(os.devnull, 'w')

sns.set(context='notebook', style='ticks', font_scale=2, font='sans-serif',  rc={"lines.linewidth": 1.2})
sns.set_style("white")
sns.set(style="whitegrid")
sns.set_style("ticks", {"xtick.major.size": 2, "ytick.major.size": 2})
sns.despine(left=True, bottom=True)
bg_color = (0.88,0.85,0.95)
bg_color = (1, 1, 1)
bg_color = 'white'
plt.rcParams['figure.facecolor'] = bg_color
plt.rcParams['axes.facecolor'] = bg_color
plt.rcParams["savefig.facecolor"] = bg_color
COLOR = 'black'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
plt.rcParams.update({"savefig.format": 'png'})


max_width = 1800
padding_top = 10
padding_right = 5
padding_left = 5
padding_bottom = 10
COLOR = 'black'
BACKGROUND_COLOR = 'white'
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )

# axis=0 sum over features=1;
# axis=1 sum over groups=1; relative feature importance change (0-1) for every group

import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, filters, measure
import nibabel as nib
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.express as px

st.write("# Exploring structural brain MRI changes with Ageing")

st.write("## Part 1: XGBoost Predictive Model")
st.write("XBoost model is trained using T1 brain imaging features and the outcome is MOCA or MDS-UPDRSIII score. Since it is a continuous variable we converted into ordered classes and then trained a classification model. The analysis is done on 3D T1 MRI images from PPMI and ADNI cohorts. The average importance score and list of features used are at [link](https://drive.google.com/drive/folders/1vj2xGc8YB0RbJ-uBQ93QcYS0k2GQRP1b?usp=sharing).")
cols = st.columns(2)
cols[0].write('### MOCA Prediction Performance')
cols[0].image('moca.png', width=450 )
cols[1].write('### MDS-UPDRSIII Prediction Performance')
cols[1].image('updrs.png', width=450)

st.write("## Part 2: Model Interpretability")

st.write("***SHAP*** [[link](https://christophm.github.io/interpretable-ml-book/shapley.html)] assigns an attribution score to each combination of a feature and an input sample, such that a high attribution score indicates that this feature plays an important role in the prediction for that sample. For the interpretation analysis, in this dashboard we can choose the ***prediction task*** (i.e. MOCA/UPDRS) score. The generated figures will show the importance of each brain region for subjects within different age groups. The importance can be intepreted as the impact on the model output (in our case it is the probability of lower moca score (0-24) or higher MDS-UPDRS PartIII score (>15) ).  We can groupby other factors such as 'APOE4 Mutation' to see the disease progression for different groups. We can select the type of ***normalization*** for feature importance scores.")


st.write ("#### Select a 3D image slice")
@st.cache
def plot_plotly_slice(slice_number, axis_number=0):
    fpath2 = "LabeledImages/OASIS-TRT-20_DKT31_CMA_labels_in_MNI152_v2/OASIS-TRT-20-1_DKT31_CMA_labels_in_MNI152.nii.gz"
    segmented_image =  nib.load(fpath2).get_fdata()
    if axis_number == 0:
        numpy_image = segmented_image[slice_number, :, :]
    elif axis_number==1:
        numpy_image = segmented_image[:, slice_number, :]
    else:
        numpy_image = np.rot90(segmented_image[:, :, slice_number])
    image = numpy_image
    image_label = image.astype(np.int32)
    
    prop_table = measure.regionprops_table(
        image_label, intensity_image=image, 
        properties=["label", "area", "perimeter", "mean_intensity"]
    )
    table = pd.DataFrame(prop_table)
    table['label'] = table['label'].map(int)
    
    mapping_index = pd.read_csv('feature_mapping.csv')
    replacement_dict1 = dict(zip(mapping_index['region_index'].map(int), mapping_index['region_name']))
    replacement_dict2 = dict(zip(mapping_index['region_index'].map(int), mapping_index['symbol']))
    table['region_name'] = table['label'].map(lambda x: replacement_dict1.get(x, np.nan))
    table['label'] = table['label'].map(lambda x: x if x in replacement_dict1 else np.nan)
    table = table[['region_name', 'label']]
    table = table.dropna()
    table['label'] = table['label'].map(int)
    


    fig = px.imshow(image, binary_string=True, binary_backend="jpg",)
    
    for rid, row in table.iterrows():
        label = row.label
        contour = measure.find_contours(image_label == label, 0.5)[0]
        y, x = contour.T
        # Now we add custom hover labels to display the computed region properties
        hoverinfo = "<br>".join([f"{val}" 
                                 for col, val in row.iteritems() if col=='region_name'])
        
        fig.add_scatter(
            x=x,
            y=y,
            mode="lines",
            fill="toself",
            showlegend=False,
            hovertemplate=hoverinfo,
            hoveron="points+fills",
        )
    
    fig.update_layout(
    autosize=True,
    width=400,
    height=400,
    title={
            'text' : f"axis_number={axis_number};slice_number={slice_number}",
            'x':0.5,
            'xanchor': 'center'
        }
    ) 
    return fig

axis_number = 1
slice_number = 120
st.write("Note: For interpretation, we will observe features importance score of different brain regions on a particular segmented image slice. Select the axis/slice number below. Check on the checkbox to see the available slices for better visualization.")
if st.checkbox("View Available Slices"):
    with st.container():
        cols = st.columns(3)
        cols[0].plotly_chart( plot_plotly_slice(60, axis_number=0) )
        cols[0].plotly_chart( plot_plotly_slice(60, axis_number=1) )
        cols[0].plotly_chart( plot_plotly_slice(60, axis_number=2) )
        cols[1].plotly_chart( plot_plotly_slice(90, axis_number=0) )
        cols[1].plotly_chart( plot_plotly_slice(90, axis_number=1) )
        cols[1].plotly_chart( plot_plotly_slice(90, axis_number=2) )
        cols[2].plotly_chart( plot_plotly_slice(120, axis_number=0) )
        cols[2].plotly_chart( plot_plotly_slice(120, axis_number=1) )
        cols[2].plotly_chart( plot_plotly_slice(120, axis_number=2) )
 
cols = st.columns(2)
axis_number = cols[0].selectbox("Select an axis along which slicing should be done", [0, 1, 2], index=1)
slice_number = cols[0].selectbox("Select a slice number for the chosen axis", [60, 90, 120], index=2)
cols[1].write('#### Segmented MRI slice image for interpretation')
cols[1].plotly_chart( plot_plotly_slice(slice_number, axis_number=axis_number)  )
all_images = os.listdir('my_image_progression')

st.write("### Visualize Feature Importance Score (brain regions)")
cols = st.columns(3)
list_prediction_tasks = ['GPU_MOCA_3GROUPS_PREDICTION', 'GPU_MDS-UPDRSPartIII_2GROUPS_PREDICTION' ]# 'GPU_DIAGNOSIS_PREDICTION']
# list_prediction_tasks = list(set([col.split('&')[0] for col in all_images]))
show_actual = {
  'Lower MOCA score (worse outcome)': f'{axis_number}&{slice_number}&GPU_MOCA_3GROUPS_PREDICTION', 
  'Higher MDSUPDRS score (worse outcome)': f'{axis_number}&{slice_number}&GPU_MDS-UPDRSPartIII_2GROUPS_PREDICTION', 
}
prediction_task = cols[0].selectbox('Select the prediction task', list(show_actual.keys()))
prediction_task = show_actual[prediction_task]

groups = {
   'SELECT': 'SELECT',
   'LRRK2 Mutation': 'LRRK2 Mutation',
   "APOE4 Mutation": "APOE4 Mutation",
   "GBA Mutation": "GBA Mutation",
   'STUDY': 'STUDY',
   'DIAGNOSIS': 'DIAGNOSIS',
} 
group = cols[1].selectbox('Select the groupby feature', list(groups.keys()))
group = groups[group] 

shap_values_type_dict = {
    'adjusted value (adjusting for risk across groups)' : 'relativemean_mean_df',
    'raw value' : 'mean_mean_df',
}
shap_values_type = cols[2].selectbox('Select the SHAP value type', list(shap_values_type_dict.keys()))
shap_values_type = shap_values_type_dict[shap_values_type] 



norm_axis = {
    # "Normalized within Groups -> Minmax across groups": f"{prediction_task}&{group}&{shap_values_type}_norm_axis1_minmax_axis0.png", 
    "Unnormalized": f"{prediction_task}&{group}&{shap_values_type}_unnormalized_data.png", 
    # "Min-Max Normalize Longitudinal": f"{prediction_task}&{group}&{shap_values_type}_minmax_axis0.png" ,	
    # "Min-Max Normalize Cross-Sectional": f"{prediction_task}&{group}&{shap_values_type}_minmax_axis1.png", 	
    # "Z-score Normalize Longitudinal": f"{prediction_task}&{group}&{shap_values_type}_norm_axis0.png", 	
    # "Z-score Normalize Cross-Sectional": f"{prediction_task}&{group}&{shap_values_type}_norm_axis1.png" 	
}

normalization = 'Unnormalized'# cols[3].selectbox('Select the normalization type', list(norm_axis.keys()))
normalization = norm_axis[normalization]


# + list(set([col.split('&')[1] for col in all_images if col.split('&')[0]==prediction_task and (not 'SELECT' in col)]))
# image_name = f"image_progression/{prediction_task}_.png" 
# st.image(image_name, use_column_width=True)
# class_list  = list(set([col.split('&')[2].split('.')[0] for col in all_images if col.split('&')[0]==prediction_task and col.split('&')[1]==group]))
# class_name = st.selectbox('Select the class/score-range', sorted(class_list))
# image_name = f"my_image_progression/{prediction_task}&{group}&{class_name}.png" 
image_name = f"my_image_progression/{normalization}" 

st.image(image_name, use_column_width=True)
st.write('***Note***: In the figure above,  *higher valued (red color)* feature indicates that this feature causes larger increase in predicted probability towards the worse outcome (lower MOCA / higher UPDRS) as compared to features having *lower value (blue color)*.')
st.write('---')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator



prediction_task = '_'.join(prediction_task.split('&')[-1].split('_')[1:-1])
if prediction_task == 'MOCA_3GROUPS':
    classname = "-1-24"
elif prediction_task == 'MDS-UPDRSPartIII_2GROUPS':
    classname = "15-95"

@st.cache
def get_data(prediction_task, group):
    # data = pd.read_csv('training.csv')
    # metadata = pd.read_csv('all_outputs.csv')
    # A = data.set_index('image_date_unique_id')
    # B = metadata.set_index('image_date_unique_id')
    # Z = A.merge(B, left_index=True, right_index=True)
    # Z['SELECT'] = ['A'] * len(Z)
    Z = pd.read_parquet('imaging_data.gziprs')
    A = Z.dropna(subset=[prediction_task, 'AGE'])
    A['AGE_6GROUPS'] = list(pd.qcut(A['AGE'], 6, retbins=True, precision=0, duplicates='drop')[0].map(lambda i: f"{(int(i.left))}-{(int(i.right))}"))
    all_feature_list = Z.columns[Z.columns.str.contains('InvicroT1@')]
    g = list(A.groupby([group]))
    return g, all_feature_list

cols = st.columns(2)
g, all_feature_list = get_data(prediction_task, group)
cols[0].write("### Feature Value Visualization")
cols[0].write("Here, we can how actual feature value changes for different classes (MOCA/UPDRS) with age. The plot also shows the significance measure for every age range.")
feature_mapping_actual = { i.split('@')[-1]:i for i in all_feature_list }
cols[1].write('\n\n')
feature = cols[1].selectbox('Select the feature for box plot', list(feature_mapping_actual.keys()))

feature = feature_mapping_actual[feature]
r_feature_mapping = {j:i for i,j in feature_mapping_actual.items()}


# plt.rcParams.update({'font.size': 6})

# fig, axs = plt.subplots(len(g), 1, squeeze=False, figsize=(8*len(g), 4.5*len(g)))
# axslist = axs.reshape(-1)
for i in range(len(g)):
    K = g[i][1].copy()
    if g[i][0] in ['GENPD', 'GENUN', 'SWEDD', 'PRODROMA']:
        continue
    if 'UPDRS' in prediction_task and g[i][0] == 'Control':
        continue
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    K[prediction_task] = K[prediction_task].map(lambda x: x if x==classname else "Other")
    K = K.dropna(subset=[prediction_task, feature, 'AGE_6GROUPS'])
    significanceComparisons = []
    for x in sorted(K['AGE_6GROUPS'].unique()):
        l =[]
        for z in sorted(K[prediction_task].unique()):
            l.append((x, z))
        significanceComparisons.append(l)
    fig_args = {
            'x': "AGE_6GROUPS",
            'y': feature,
            'hue':prediction_task,
            'data': K,
            'dodge': True,
            'linewidth': 1,
            'fliersize': 1,
            'order':  sorted(K['AGE_6GROUPS'].unique())

    }
    configuration = {'test':'Mann-Whitney',
                 'comparisons_correction':None,
                 'text_format':'star'}
    sns.boxplot(ax=ax,**fig_args)
    ax.set(ylabel=r_feature_mapping[feature])
    annotator = Annotator(ax=ax, pairs=significanceComparisons, **fig_args)
    annotator.configure(**configuration).apply_test().annotate()
    plt.legend(loc='upper right', ncol=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('' if g[i][0]=='A' else g[i][0])# , fontsize=12)
    ax.set_xlabel('Age Range')# , fontsize=10)
    ax.set_ylabel(r_feature_mapping[feature])#  , fontsize=8)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.legend(bbox_to_anchor=(.90, 1.1), ncol=2, title=prediction_task, fontsize=8, title_fontsize=8)


    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png")
    cols[i%2].image(buf, width=800)
# st.pyplot(fig, bbox_inches='tight')




# data = pd.read_csv('training.csv')
# metadata = pd.read_csv('all_outputs.csv')
# A = data.set_index('image_date_unique_id')
# B = metadata.set_index('image_date_unique_id')
# 
# 
# import matplotlib.pyplot as plt
# import seaborn as sns
# Z = A.merge(B, left_index=True, right_index=True)
# A = Z.dropna(subset=['_'.join(prediction_task.split('_')[1:-1])])
# A['AGE_6GROUPS'] = pd.qcut(A['AGE'], 6, retbins=True, precision=0, duplicates='drop')[0].map(lambda i: f"{(int(i.left))}-{(int(i.right))}")
# selected_cols = ['ID:InvicroT1@left superior temporal_vol_dktregions',]
# g = list(Z.groupby([group]))
# for col in selected_cols:
#     sns.boxplot(y=col, hue="MOCA_3GROUPS", x='AGEX', data=A)
#     plt.show()
# 
# 
