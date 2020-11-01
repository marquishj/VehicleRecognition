# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 16:49
# @Author  : Hou Jue
# @Email   : 382062104@qq.com
# @File    : all_combination_3.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# import numpy as np
import numpy as np
from sklearn.cluster import KMeans
# import sklearn.preprocessing.MinMaxScaler
# import sklearn.preprocessing.StandardScaler
# import sklearn
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
# -*- coding: utf-8 -*-
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def cal(item1,features_label,data,x,accuracy,si_scores):
    acc_0=[]
    # si_score=[]
    if(x==item1.shape[1]):
        # if item2 == [] or item2==[[]]:
        if data == []:
           return 0
        combination,acc_0,si_score,accuracy,si_scores=k_means(data,accuracy,si_scores)
        # dbscan(data)

    else:
        item3=data[:]
        data.append(item1[:,x])
        cal(item1,features_label,data,x+1,accuracy,si_scores)
        cal(item1,features_label,item3,x+1,accuracy,si_scores)
    return acc_0,si_scores,accuracy

def get_keys(features_label, value):
    return [k for k, v in features_label.items() if v == value]

def findBestReuslt(accuracy,si_scores):
    bestAccuracy=max(accuracy)
    bestSi_score=max(si_scores)
    return bestAccuracy,bestSi_score


def findBestAccuracy(accuracy):
    bestAccuracy=max(accuracy)
    # bestSi_score=max(si_scores)
    return bestAccuracy

def findBestSi(si_scores):
    # bestAccuracy=max(accuracy)
    bestSi_score=max(si_scores)
    return bestSi_score

'dbsacn'
def dbscan(item2):
    y_pred = DBSCAN().fit_predict(item2)
    plt.scatter(item2[:, 0], item2[:, 1], c=y_pred)
    plt.show()


'k_means方法'
def k_means(item2,accuracy,si_scores):
    combination = []
    item2=np.array(item2)
    kmeans = KMeans(
        n_clusters=2,
        random_state=0
    ).fit(item2.T)
    label = kmeans.labels_
    print(label)
    label = list(label)
    acc_0 = accuracy_0(label)
    acc_1 = accuracy_1(label)
    for i in range(len(item2)):
        combination.append(get_keys(features_label, item2[i].tolist()))
    print('feature combination: {}'.format(combination))
    print('accuracy_0: {}'.format(acc_0))
    print('accuracy_1: {}'.format(acc_1))
    # 使用轮廓系数评估模型的优虐
    si_score = silhouette_score(item2.T, kmeans.labels_,
    metric='euclidean', sample_size=len(item2.T))
    print('si_score: {:.4f}'.format(si_score))
    accuracy.append(acc_0)
    si_scores.append(si_score)
    return combination,acc_0,si_score,accuracy,si_scores

def accuracy_0(label):
    n=0
    m=0
    label_0=list([0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0])
    for i in label:
        if i == label_0[n] :
            m = m + 1
        n=n+1
    acc_0=m/25
    return acc_0

def accuracy_1(label):
    m=0
    n=0
    label_1=list([0,1,1,0,0,2,1,2,2,0,2,1,0,0,0,2,2,1,1,0,0,1,0,1,1])
    for i in label:
        if i == label_1[n]:
            m = m + 1
        n=n+1
    acc_1=m/25
    return acc_1





if __name__ == '__main__':

    accuracy=[]
    si_scores=[]

    '''BEV特征提取'''
    BEV_daily_distance_mean =[-1.1523545528648889, 0.3417909464676774, -0.5419080456980253, 1.1187622362996381, 1.9448269807577925, -0.9770048538811394, -0.23281352349865447, -1.4014134967626322, 3.7494940074544605, -0.7931827523206774, -0.9671281636202758, 1.3540817664040043, 2.1129102086262757, 2.9880692997727865, 2.3978807180945862, 2.564093856611087, -1.2901260533100944, 0.20778877432236936, -0.0146171799133254, 3.989902930516838, 0.7752243887468431, 1.395807361450496, 2.3670202812923784, 0.003340333799245773, 1.7399818215741631] # 统计日均里程
    BEV_daily_distance_std =[-1.121127953056298, -0.22191162989002292, -0.468647606749319, -0.28030319496091555, 0.7204065530370907, -0.7349532084250934, -0.14976360984308063, -0.9367809324764715, 0.5697283077699826, -0.9592326273449059, -0.7167543855403666, 0.33066222377753424, 0.7341821877123338, 1.0718851635420472, 1.0131555443231461, 0.9216857157743186, -1.4734884939394577, -0.20334516267975986, -0.021366783896013156, 0.2320635513896065, 0.16813904622138176, -0.5925568368014714, 0.7748890123786099, -0.8029534036523243, -0.3068033412910889]# 统计日里程标准差
    BEV_night_distance_mean =[-0.38279737523171936, -0.18877867292385378, 1.887630368153721, 3.9859381375315355, 3.70557802277213, 1.403629445137867, 0.792174347304973, -0.2883493244294438, 1.6157718040868823, -0.06316955677712471, 0.7691144197288264, 0.8495305457187723, 0.3602425259087915, 0.5552542789012848, 5.414239603907824, 7.054833795547546, -0.5671398774160025, 1.1881217869873701, 2.079287188705736, 3.2953367524524553, -0.12581768336045682, -0.7953076985836527, 2.384195374985208, 1.2078991229050842, 0.4009701914550498]##统计夜间日均里程
    BEV_night_distance_percentage = [-0.6097479158141259, 0.18554390593060038, 3.1239040169286625, 2.0535438826408785, 1.9207404266792585, -0.4679169934273931, 1.4433875075256246, 0.22081488090502327, -0.3235003949671269, 0.4558216081250091, 0.9214008141462728, 1.049795493170281, -0.27626488471537297, -0.15515333191127184, 3.2858367127691763, 3.0831164955466797, -0.5414635046633602, 0.020347296043003622, 2.472715285644744, 1.9849226171941476, -0.1551759638446311, -0.6570447267126164, 1.3500314813711343, 0.03764046982312223, 0.31665171977784795]# 统计夜间里程占比 0：00-5：00
    BEV_am_peak_percentage =  [1.6762090016331026, -0.271133635333542, -0.48116264744231696, -1.4755008553091382, -0.6904573980340397, 1.9685927062719513, -0.29664647220676593, -0.4457490731052066, -0.17179507497679908, -0.11189815626083206, 0.0959650108286484, -1.5472279473326054, -0.32924359881344056, -1.349437756120087, -0.5051715087647763, -0.5511011678136065, 0.6800758751167316, 0.11133231512433356, -0.37602106300731153, -0.7048634015171718, 2.0998165547776972, 1.789287966057264, -0.8946653110612125, 2.0968302164391455, -0.9319668548787869]# 统计早高峰里程占比  7：00-9：00
    BEV_pm_peak_percentage =[0.7996280174243483, -0.8048443041004449, -0.48498825357104214, 0.11390603291455531, -0.8173008190588968, -0.464410662300304, -0.5735902201751654, -0.8441375197542054, -0.3907343895899505, -0.758897594745007, 0.13101542127070026, 0.42781738841141786, -0.3379303063569493, 0.047443648187285414, -0.4587927865383168, -1.8766850272381612, 1.9019878226537612, -0.46968632245373876, -0.8139861435355585, -0.3576849018159679, 0.28902371279574673, 0.5395477044266084, -0.47321962195650086, -0.4563766786157817, -1.3518279681539043]# 统计晚高峰里程占比  17：00-19：00
    BEV_weekends_distance_percentage = [-0.015450791550570782, -0.8458464457109423, 1.0705612256216748, -0.1832508827074612, -0.22619739198906091, -1.8624568982513965, 1.1909809253746615, 0.9345647840030432, 0.3953640372257148, -0.8245852606809618, 0.4265334598332262, -0.785248512047222, -0.1301555389696291, -0.922629641075733, 0.18248042635726536, -0.8340586014148531, -0.637740632074162, -0.25028026663742714, 0.9710708062019482, 0.38927733436576306, -0.711675734280905, 0.22235567351206775, 0.1359355959692118, -1.1826490603180588, 0.19508668651207123] # 统计周末里程占比
    BEV_charging_rate_mean = [-1.6746657746630724, -0.59296077860366, 1.1078778192581378, 8.327713631147882, 5.710834734584417, -0.35743511976652914, -1.273790461282037, -0.7688385101067011, 12.79700399445406, -1.553723838932599, -1.6023438422660947, 1.0867740172665652, 6.659081747208161, 9.82331533018679, 6.102888033422787, 8.166476770489563, -0.4338924978039777, -0.6158262162261432, -1.0456261442949946, 6.683048627875319, -1.7362787628926895, 5.976709170055356, 5.83794343550155, -0.6063822220964908, 0.7760503581971758] #  # 统计充电速率
    BEV_chains_num=[-2.5235300239955802, -2.1269472620980356, -1.873389167723901, -2.1039102983437377, -1.6394052175542948, -2.5100683786401348, -1.8155106364922289, -2.801443805212215, -1.5557157806027522, -2.4543164700706406, -2.5128865174809007, -1.9556724446204325, -1.7767414117950833, -1.8067713645610752, -1.649355247775945, -1.4177879583038655, -2.9701807117205434, -1.7542947804346452, -1.68505823300582, -1.3253292148430122, -1.9236232931681843, -1.7352009748083106, -1.434344523993367, -2.254142049031015, -1.7192875373526548]
    BEV_chains_mile_std= [0.0794946842791683, 0.35703641930377145, 0.22359937528609367, 1.0600562966652043, 0.7541346724435001, 0.6138725160752863, 0.40002581583602886, -0.14553081455170883, 2.416274138020278, 0.08405243069650373, 0.5784288937432829, 1.4035546866188733, 0.5605336045082464, 1.830346453490036, 0.6795116069131096, 1.085307610032655, 23.23840565370281, 0.6124813852425322, 0.25450634359622976, 0.9275919057822349, 0.5496078383665994, 0.22945376163838913, 0.5509911468055734, 0.10074542267354984, 0.4398164454290323]
    BEV_daily_num_run_status= [-0.18768744976339943, 0.9250461479653876, -0.11205511946928258, 2.089453681465247, 2.1093323520484604, -0.38585461130936755, 0.09615105243100636, -0.46968823431524925, 1.8565140304137246, -0.0584966834088614, -0.3314491019319586, 1.3812992710579057, 2.8765936736046362, 1.7568449367915342, 3.414383571891162, 1.8143621431482213, -0.4285910063534489, 0.15996998391060308, 0.15909043339231455, 4.026503029539181, 0.359570422232921, 1.0749031810635512, 2.3681990784870384, 0.09574059838483504, 2.163302665027001]

    BEV_daily_distance_mean=[]
    BEV_daily_distance_std=[]
    BEV_night_distance_mean=[]
    BEV_night_distance_percentage=[]
    BEV_am_peak_percentage=[]
    BEV_pm_peak_percentage=[]
    BEV_weekends_distance_percentage=[]
    BEV_charging_rate_mean=[]
    BEV_chains_num=[]
    BEV_chains_mile_std=[]
    BEV_daily_num_run_status=[]


    features = np.array([BEV_daily_distance_mean, \
                  BEV_daily_distance_std, \
                  BEV_night_distance_mean, \
                  BEV_night_distance_percentage, \
                  BEV_am_peak_percentage, \
                  BEV_pm_peak_percentage, \
                  BEV_weekends_distance_percentage, \
                  BEV_charging_rate_mean,
                  BEV_chains_num, \
                  BEV_chains_mile_std, \
                  BEV_daily_num_run_status,\
                  ])

    features_label = {0: BEV_daily_distance_mean,
                      1: BEV_daily_distance_std,
                      2: BEV_night_distance_mean,
                      3: BEV_night_distance_percentage,
                      4: BEV_am_peak_percentage,
                      5: BEV_pm_peak_percentage,
                      6: BEV_weekends_distance_percentage,
                      7: BEV_charging_rate_mean,
                      8: BEV_chains_num,
                      9: BEV_chains_mile_std,
                     10: BEV_daily_num_run_status
                      }
    feature_label = [features_label[i] for i in range(len(features))]
    feature_label=np.array(feature_label)
    acc_0,si_score,accuracy=cal(feature_label.T,features_label,[],0,accuracy,si_scores)
    print('----------------------------------------')
    print('The best accuracy: {}'.format(findBestAccuracy(accuracy)))
    print('----------------------------------------')
    print('The best si_score: {}'.format(findBestSi(si_score)))



