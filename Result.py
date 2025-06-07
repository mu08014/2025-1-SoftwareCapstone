import matplotlib.pyplot as plt
import os

save_dir = './Result'

def H1TestAcc():
    file_name = 'h1_test_accuracy_comparison.png'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, file_name)
    epochs = list(range(1, 11))
    

    plt.figure(figsize=(10, 6))
    
    LeNet_acc = [
        0.2460000067949295,
        0.33399999141693115,
        0.4259999990463257,
        0.4020000100135803,
        0.5320000052452087,
        0.5879999995231628,
        0.6660000085830688,
        0.7160000205039978,
        0.7519999742507935,
        0.8080000281333923
    ]

    First_Replace_acc = [
        0.23199999332427979,
        0.3400000035762787,
        0.47600001096725464,
        0.5860000252723694,
        0.6679999828338623,
        0.7319999933242798,
        0.7900000214576721,
        0.8199999928474426,
        0.8600000143051147,
        0.8999999761581421
    ]
    
    Second_Replace_acc = [
        0.19200000166893005,
        0.18799999356269836,
        0.17599999904632568,
        0.1679999977350235,
        0.1979999989271164,
        0.20600000023841858,
        0.21400000154972076,
        0.17599999904632568,
        0.17000000178813934,
        0.20200000703334808
    ]

    Third_Replace_acc = [
        0.1979999989271164,
        0.21199999749660492,
        0.1899999976158142,
        0.16599999368190765,
        0.20399999618530273,
        0.1940000057220459,
        0.18000000715255737,
        0.1979999989271164,
        0.1899999976158142,
        0.1940000057220459
    ]
    Fourth_Replace_acc = [
        0.1979999989271164,
        0.17599999904632568,
        0.20800000429153442,
        0.22599999606609344,
        0.18799999356269836,
        0.17000000178813934,
        0.19200000166893005,
        0.20000000298023224,
        0.20000000298023224,
        0.20000000298023224
    ]

    
    plt.plot(epochs, LeNet_acc, marker='o', label='LeNet')
    plt.plot(epochs, First_Replace_acc, marker='s', label='First Replace')
    plt.plot(epochs, Second_Replace_acc, marker='^', label='Second Replace')
    plt.plot(epochs, Third_Replace_acc, marker='d', label='Third Replace')
    plt.plot(epochs, Fourth_Replace_acc, marker='x', label='Fourth Replace')

    plt.title('test Accuracy Comparison per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('test Accuracy')
    plt.xticks(range(1, 11))
    plt.ylim(0.00, 1.00)
    plt.grid(True)
    
    plt.legend(title='Models', fontsize=10, title_fontsize=11, loc='lower right')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"그래프 저장 완료: {save_path}")
    
def H1TestLoss():
    file_name = 'h1_test_loss_comparison.png'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, file_name)
    epochs = list(range(1, 11))
    

    plt.figure(figsize=(10, 6))
    
    LeNet_loss = [
        1.606155276298523,
        1.5660761594772339,
        1.50639808177948,
        1.4142069816589355,
        1.223501205444336,
        1.0721304416656494,
        0.9164761304855347,
        0.7780568599700928,
        0.6785505414009094,
        0.5895968079566956
    ]

    First_Replace_loss = [
        1.6277503967285156,
        1.5411884784698486,
        1.4160765409469604,
        1.220739722251892,
        0.9994291663169861,
        0.7880991697311401,
        0.6039595007896423,
        0.4777946472167969,
        0.39942482113838196,
        0.29977673292160034
    ]
    
    Second_Replace_loss = [
        1.719197392463684,
        1.6384248733520508,
        1.6215236186981201,
        1.6142070293426514,
        1.60990309715271,
        1.609136700630188,
        1.6113083362579346,
        1.6103200912475586,
        1.61387038230896,
        1.6111382246017456
    ]

    Third_Replace_loss = [
        2.2206265926361084,
        1.7528090476989746,
        1.6626396179199219,
        1.6370729207992554,
        1.6280395984649658,
        1.626970648765564,
        1.6245946884155273,
        1.614023208618164,
        1.6182971000671387,
        1.6140971183776855
    ]
    
    Fourth_Replace_loss = [
        2.224972724914551,
        1.852192997932434,
        1.6768934726715088,
        1.6126525402069092,
        1.61767578125,
        1.6111806631088257,
        1.6096845865249634,
        1.6098814010620117,
        1.6094486713409424,
        1.6094480752944946
    ]

    
    plt.plot(epochs, LeNet_loss, marker='o', label='LeNet')
    plt.plot(epochs, First_Replace_loss, marker='s', label='First Replace')
    plt.plot(epochs, Second_Replace_loss, marker='^', label='Second Replace')
    plt.plot(epochs, Third_Replace_loss, marker='d', label='Third Replace')
    plt.plot(epochs, Fourth_Replace_loss, marker='x', label='Fourth Replace')

    plt.title('test Loss Comparison per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('test Loss')
    plt.xticks(range(1, 11))
    plt.ylim(0.00, 2.30)
    plt.grid(True)
    
    plt.legend(title='Models', fontsize=10, title_fontsize=11, loc='lower right')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"그래프 저장 완료: {save_path}")
    
def H2TestAcc():
    file_name = 'h2_test_accuracy_comparison.png'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, file_name)
    epochs = list(range(1, 11))
    

    plt.figure(figsize=(10, 6))

    First_Replace_acc = [
        0.23199999332427979,
        0.3400000035762787,
        0.47600001096725464,
        0.5860000252723694,
        0.6679999828338623,
        0.7319999933242798,
        0.7900000214576721,
        0.8199999928474426,
        0.8600000143051147,
        0.8999999761581421
    ]
    
    FS_Replace_acc =  [
        0.33399999141693115,
        0.5419999957084656,
        0.6800000071525574,
        0.7300000190734863,
        0.8040000200271606,
        0.8240000009536743,
        0.8679999709129333,
        0.8679999709129333,
        0.8820000290870667,
        0.8840000033378601
    ]

    FST_Replace_acc = [
        0.25999999046325684,
        0.4620000123977661,
        0.6639999747276306,
        0.7179999947547913,
        0.7239999771118164,
        0.7839999794960022,
        0.8519999980926514,
        0.8420000076293945,
        0.8679999709129333,
        0.8820000290870667
    ]
    
    FSTF_Replace_acc = [
        0.20200000703334808,
        0.23399999737739563,
        0.31200000643730164,
        0.38999998569488525,
        0.44200000166893005,
        0.414000004529953,
        0.4880000054836273,
        0.4860000014305115,
        0.5299999713897705,
        0.5339999794960022
    ]

    
    plt.plot(epochs, First_Replace_acc, marker='s', label='One Replace')
    plt.plot(epochs, FS_Replace_acc, marker='^', label='Two Replace')
    plt.plot(epochs, FST_Replace_acc, marker='d', label='Three Replace')
    plt.plot(epochs, FSTF_Replace_acc, marker='x', label='Four Replace')

    plt.title('test Accuracy Comparison per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('test Accuracy')
    plt.xticks(range(1, 11))
    plt.ylim(0.00, 1.00)
    plt.grid(True)
    
    plt.legend(title='Models', fontsize=10, title_fontsize=11, loc='lower right')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"그래프 저장 완료: {save_path}")
    
def H2TestLoss():
    file_name = 'h2_test_loss_comparison.png'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, file_name)
    epochs = list(range(1, 11))
    

    plt.figure(figsize=(10, 6))

    First_Replace_loss = [
        1.6277503967285156,
        1.5411884784698486,
        1.4160765409469604,
        1.220739722251892,
        0.9994291663169861,
        0.7880991697311401,
        0.6039595007896423,
        0.4777946472167969,
        0.39942482113838196,
        0.29977673292160034
    ]
    
    FS_Replace_loss = [
        1.537671685218811,
        1.262542724609375,
        0.974952757358551,
        0.793014645576477,
        0.617601752281189,
        0.5081566572189331,
        0.4274722933769226,
        0.3809369206428528,
        0.33347830176353455,
        0.31624481081962585
    ]

    FST_Replace_loss = [
        1.8484355211257935,
        1.376361608505249,
        1.1275608539581299,
        0.9174788594245911,
        0.8103311657905579,
        0.6324349641799927,
        0.5144414901733398,
        0.4753747582435608,
        0.3889445662498474,
        0.36964768171310425
    ]
    
    FSTF_Replace_loss = [
        2.0453453063964844,
        1.6723933219909668,
        1.5705543756484985,
        1.4456106424331665,
        1.4035323858261108,
        1.4007426500320435,
        1.3419744968414307,
        1.322704792022705,
        1.2711488008499146,
        1.2330405712127686
    ]

    
    plt.plot(epochs, First_Replace_loss, marker='s', label='One Replace')
    plt.plot(epochs, FS_Replace_loss, marker='^', label='Two Replace')
    plt.plot(epochs, FST_Replace_loss, marker='d', label='Three Replace')
    plt.plot(epochs, FSTF_Replace_loss, marker='x', label='Four Replace')

    plt.title('test Loss Comparison per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('test Loss')
    plt.xticks(range(1, 11))
    plt.ylim(0.00, 2.30)
    plt.grid(True)
    
    plt.legend(title='Models', fontsize=10, title_fontsize=11, loc='lower right')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"그래프 저장 완료: {save_path}")
    
def H1ParamCount():
    file_name = 'h1_param_count.png'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, file_name)
    
    model_names = [
        "LeNet", "First Replace", "Second Replace", "Third Replace", "Fourth Replace"
    ]

    param_counts = [
        139493, 139173, 132853, 158389, 102565
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, param_counts)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:,}', 
                ha='center', va='bottom')

    plt.title("Parameter Count per Model")
    plt.xlabel("Model")
    plt.ylabel("Number of Parameters")
    plt.ylim(min(param_counts)-10000, max(param_counts) + 10000)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"그래프 저장 완료: {save_path}")
    
def H1ParamCount():
    file_name = 'h2_param_count.png'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, file_name)
    
    model_names = [
        "One Replace", "Two Replace", "Three Replace", "Four Replace"
    ]

    param_counts = [
        139173, 129925, 111429, 74501
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, param_counts)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:,}', 
                ha='center', va='bottom')

    plt.title("Parameter Count per Model")
    plt.xlabel("Model")
    plt.ylabel("Number of Parameters")
    plt.ylim(min(param_counts)-10000, max(param_counts) + 10000)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"그래프 저장 완료: {save_path}")
    
if __name__=='__main__':
    H1ParamCount()
