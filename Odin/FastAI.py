import pandas as pd
from random import sample
from fastai.tabular.all import *
from sklearn.model_selection import train_test_split
# train, test = train_test_split(sdf)
# from fastai import *

df = pd.read_csv("Odin/fused_dataset_plain.csv").sample(n = 1000)


from sklearn.decomposition import PCA









dep_var = 'choice'
leavout = ['sted_o','sted_d', 'aankpc','av_car', 'av_carp', 'av_transit', 'av_cycle', 'av_walk', 'c_carp', 'c_cycle', 'c_walk', 'c_car']
cat_names = ['ovstkaart','weekday','d_hhchildren', 'd_high_educ', 'gender', 'age', 'pur_home', 'pur_work', 'pur_busn', 'pur_other','driving_license', 'car_ownership',
       'main_car_user', 'hh_highinc10', 'hh_lowinc10', 'hh_highinc20', 'av_car', 'av_carp', 'av_transit', 'av_cycle', 'av_walk']
cont_names = ['c_car','departure_rain', 'arrival_rain','dist_car', 'dist_carp',
       'dist_transit', 'dist_cycle', 'dist_walk', 't_car', 't_carp',
              't_transit', 't_cycle', 't_walk', 'c_transit', 'vc_car', 'pc_car','actduur', 'reisduur_sec', 'afstv_m']

sdf = df[cont_names + cat_names + [dep_var]]

splits = RandomSplitter(valid_pct=0.2)(range_of(sdf))

to = TabularPandas(sdf, procs=[Categorify, FillMissing,Normalize],
                   cat_names= cat_names, cont_names= cont_names,
                   y_names= dep_var,
                   splits=splits, y_block = CategoryBlock)

dls = to.dataloaders(bs=64)
# dls.show_batch()

learn = tabular_learner(dls, metrics=accuracy, layers=[10,3], cbs = ActivationStats(every=4))
learn.lr_find()
learn.fit_one_cycle(2, learn.lr)
learn.save('nn')

learn.show_results()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
plt.show()


stattt = learn.activation_stats.stats[1]




















a = learn.model.layers[1].parameters.Linear
movie_w = learn.model.movie_factors[top_idxs].cpu().detach()

a = list(learn.model.parameters())
b = a[-1].T

learn.model.layers[0].register_forward_hook(example_forward_hook)
learn.model.layers[1].register_forward_hook(example_forward_hook)
learn.model.layers[2].register_forward_hook(example_forward_hook)
# for i in sdf.columns:
#     print(i)
#     print(len(sdf[i].unique()))

x, y, z = dls.one_batch()
x[0]
oneex = (x[0:3],y[0:3],z[0:3])
preds,_ = learn.get_preds(dl=[(x,y,z)])
preds,_ = learn.get_preds(dl=[oneex])


testinp = dls.one_batch()
# testinp.reshape()

activationstt = []#dict()
def example_forward_hook(m,i,o):
    print('hi')
    activationstt.append(m)

hook = learn.register_forward_hook(example_forward_hook)
preds,_ = learn.get_preds(dl=[oneex])
hook.remove()

activationstt[2]