import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, padding=1),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 256, kernel_size=2),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2, padding=1),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 4),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = torch.log2(x + 1) / 16
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc(x)
        return x


if __name__ == '__main__':
    cnn = CNN()
    # cnn.load_state_dict(torch.load('test01.pkl'))

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)

    data_file_name = ['robust_0815_1706_2048.npy', 'robust_0816_0110_2048.npy',
    'robust_0816_0156_4096.npy', 'robust_0816_0217_2048.npy', 'robust_0816_1138.npy',
    'robust_0816_1224.npy', 'robust_0816_1244.npy', 'robust_0816_1325.npy',
    'robust_0816_1540.npy', 'robust_0816_1630.npy', 'robust_0816_1716.npy',
    'robust_0816_1805.npy', 'robust_0816_1845.npy', 'robust_0816_1927.npy',
    'robust_0816_2023.npy', 'robust_0817_1433.npy', 'robust_0817_1450.npy']

    '''
robust_0817_1500.npy
robust_0817_1527.npy
robust_0817_1553.npy
robust_0817_1650.npy
robust_0817_1708.npy
robust_0817_1718.npy
robust_0817_1730.npy
robust_0817_1740.npy
robust_0817_1743.npy
robust_0817_1808.npy
robust_0818_0049.npy
robust_0818_0057.npy
robust_0818_0105.npy
robust_0818_0119.npy
robust_0818_0141.npy
robust_0818_1030.npy
robust_0818_1046.npy
robust_0818_1111.npy
robust_0818_1131.npy
robust_0818_1638_7108_512.npy
robust_0818_1648_56716_4096.npy
robust_0818_1654_20956_1024.npy
robust_0818_1659_14448_1024.npy
robust_0818_1703_12276_1024.npy
robust_0818_1723_70544_4096.npy
robust_0818_1746_67492_4096.npy
robust_0818_1805_40308_2048.npy
robust_0818_1842_78180_4096.npy
robust_0818_1847_28304_2048.npy
robust_0818_1851_22064_1024.npy
-a----         2021/8/18     18:57         243371 robust_0818_1857_28252_2048.npy
-a----         2021/8/18     19:05         283317 robust_0818_1905_33720_2048.npy
-a----         2021/8/18     19:15         311208 robust_0818_1915_36512_2048.npy
-a----         2021/8/18     19:27         335678 robust_0818_1927_38716_2048.npy
-a----         2021/8/18     19:47         454708 robust_0818_1947_59832_4096.npy
-a----         2021/8/18     19:50         175516 robust_0818_1950_18460_1024.npy
-a----         2021/8/18     19:57         293452 robust_0818_1957_35052_2048.npy
-a----         2021/8/18     20:00         155208 robust_0818_2000_16316_1024.npy
-a----         2021/8/18     20:04         164961 robust_0818_2004_17048_1024.npy
-a----         2021/8/18     20:11         257074 robust_0818_2011_30456_2048.npy
-a----         2021/8/18     20:15         135340 robust_0818_2015_13456_1024.npy
-a----         2021/8/18     20:22         210380 robust_0818_2022_24696_2048.npy
-a----         2021/8/18     20:42         477901 robust_0818_2042_62192_4096.npy
-a----         2021/8/18     20:51         194895 robust_0818_2051_20336_1024.npy
-a----         2021/8/18     20:59         151093 robust_0818_2059_15004_1024.npy
-a----         2021/8/18     21:08         166399 robust_0818_2108_17048_1024.npy
-a----         2021/8/18     21:12         106145 robust_0818_2112_9524_512.npy
-a----         2021/8/18     21:17         316301 robust_0818_2117_37268_2048.npy
-a----         2021/8/18     21:20         234294 robust_0818_2120_27516_2048.npy
-a----         2021/8/18     21:21          77649 robust_0818_2121_7044_512.npy
-a----         2021/8/18     21:24         151814 robust_0818_2124_16032_1024.npy
-a----         2021/8/18     21:28         185959 robust_0818_2128_19356_1024.npy
-a----         2021/8/18     21:28          34855 robust_0818_2128_2216_128.npy
-a----         2021/8/18     21:32         155860 robust_0818_2132_16252_1024.npy
-a----         2021/8/18     21:38         247305 robust_0818_2138_28360_2048.npy
-a----         2021/8/18     21:50         396814 robust_0818_2150_52500_4096.npy
-a----         2021/8/18     21:59         269828 robust_0818_2159_32360_2048.npy
-a----         2021/8/18     22:05         145457 robust_0818_2205_15208_1024.npy
-a----         2021/8/18     22:56         203936 robust_0818_2256_22052_1024.npy
-a----         2021/8/18     23:03         428371 robust_0818_2303_56876_4096.npy
-a----         2021/8/18     23:05          84303 robust_0818_2305_7472_512.npy
-a----         2021/8/18     23:07         127027 robust_0818_2307_12940_1024.npy
-a----         2021/8/18     23:08         105685 robust_0818_2308_9536_512.npy
-a----         2021/8/18     23:13         274486 robust_0818_2313_31580_2048.npy
-a----         2021/8/18     23:15         116117 robust_0818_2315_12172_1024.npy
-a----         2021/8/18     23:17         109597 robust_0818_2317_11588_1024.npy
-a----         2021/8/18     23:19         173495 robust_0818_2319_21420_2048.npy
-a----         2021/8/18     23:23         154719 robust_0818_2323_16112_1024.npy
-a----         2021/8/18     23:27         182670 robust_0818_2327_19088_1024.npy
-a----         2021/8/18     23:31         165124 robust_0818_2331_16988_1024.npy
-a----         2021/8/18     23:33          92616 robust_0818_2333_8380_512.npy
-a----         2021/8/18     23:37         154661 robust_0818_2337_16100_1024.npy
-a----         2021/8/18     23:39          60208 robust_0818_2339_5336_512.npy
-a----         2021/8/18     23:41          79087 robust_0818_2341_7256_512.npy
-a----         2021/8/18     23:43         101284 robust_0818_2343_10500_1024.npy
-a----         2021/8/18     23:47         255292 robust_0818_2347_30992_2048.npy
-a----         2021/8/18     23:49          92482 robust_0818_2349_8100_512.npy
-a----         2021/8/18     23:52         209192 robust_0818_2352_25236_2048.npy
-a----         2021/8/18     23:54         120076 robust_0818_2354_12372_1024.npy
-a----         2021/8/18     23:56          99491 robust_0818_2356_8992_512.npy
-a----         2021/8/18     23:58         132388 robust_0818_2358_14192_1024.npy
-a----         2021/8/19      0:00         115139 robust_0819_0000_12044_1024.npy
-a----         2021/8/19      0:01          73085 robust_0819_0001_6608_512.npy
-a----         2021/8/19      0:02          72596 robust_0819_0002_6328_512.npy
-a----         2021/8/19      0:05         208185 robust_0819_0005_25020_2048.npy
-a----         2021/8/19      0:07         147576 robust_0819_0007_14836_1024.npy
-a----         2021/8/19      0:11         166486 robust_0819_0011_17024_1024.npy
-a----         2021/8/19      0:16         209192 robust_0819_0016_24704_2048.npy
-a----         2021/8/19      0:18         106011 robust_0819_0018_9520_512.npy
-a----         2021/8/19      0:43         147250 robust_0819_0043_15536_1024.npy
-a----         2021/8/19      0:46         119348 robust_0819_0046_12364_1024.npy
-a----         2021/8/19      0:47          42727 robust_0819_0047_3100_256.npy
-a----         2021/8/19      0:52         420786 robust_0819_0052_55584_4096.npy
-a----         2021/8/19      0:56         163063 robust_0819_0056_16828_1024.npy
-a----         2021/8/19      0:58         105685 robust_0819_0058_9320_512.npy
-a----         2021/8/19      0:59          70745 robust_0819_0059_6536_512.npy
-a----         2021/8/19      1:04         290384 robust_0819_0104_34596_2048.npy
-a----         2021/8/19      1:07          70640 robust_0819_0107_6408_512.npy
-a----         2021/8/19      1:16         206718 robust_0819_0116_25108_2048.npy
-a----         2021/8/19      1:18         151651 robust_0819_0118_15848_1024.npy
-a----         2021/8/19      1:21         133500 robust_0819_0121_14316_1024.npy
-a----         2021/8/19      1:23         123912 robust_0819_0123_12544_1024.npy
-a----         2021/8/19      1:27         158497 robust_0819_0127_16628_1024.npy
-a----         2021/8/19      1:34         282618 robust_0819_0134_33932_2048.npy
-a----         2021/8/19      1:38         162871 robust_0819_0138_16816_1024.npy
-a----         2021/8/19      1:43         149742 robust_0819_0143_15504_1024.npy
-a----         2021/8/19      1:47         117421 robust_0819_0147_11248_512.npy
-a----         2021/8/19      1:57         274823 robust_0819_0157_31608_2048.npy
-a----         2021/8/19      2:06         231686 robust_0819_0206_27232_2048.npy
-a----         2021/8/19      2:11         122800 robust_0819_0211_12576_1024.npy
-a----         2021/8/19      2:18         157461 robust_0819_0218_16524_1024.npy
-a----         2021/8/19      2:30         254966 robust_0819_0230_30860_2048.npy
-a----         2021/8/19      2:37         138285 robust_0819_0237_14624_1024.npy
-a----         2021/8/19      2:39          45020 robust_0819_0239_3428_256.npy
-a----         2021/8/19      2:46         123778 robust_0819_0246_12528_1024.npy
-a----         2021/8/19      2:53         124593 robust_0819_0253_12560_1024.npy
-a----         2021/8/19      2:58         100929 robust_0819_0258_9588_512.npy
-a----         2021/8/19      3:05         118044 robust_0819_0305_12152_1024.npy
-a----         2021/8/19      3:18         229730 robust_0819_0318_27144_2048.npy
robust_0819_0337_34496_2048.npy
robust_0819_0344_12200_1024.npy
robust_0819_0354_14752_1024.npy
robust_0819_0403_13744_1024.npy
robust_0819_0419_26412_2048.npy
robust_0819_0431_16380_1024.npy
robust_0819_0453_34176_2048.npy
robust_0819_0502_12180_1024.npy
robust_0819_0518_19048_1024.npy
robust_0819_0531_16608_1024.npy
robust_0819_0544_15368_1024.npy
robust_0819_0549_4404_256.npy
robust_0819_0605_21772_2048.npy
robust_0819_0620_15776_1024.npy
robust_0819_0634_15700_1024.npy
robust_0819_0641_6604_512.npy
robust_0819_0656_16564_1024.npy
robust_0819_0722_30852_2048.npy
robust_0819_0739_16844_1024.npy
robust_0819_0805_28208_2048.npy
robust_0819_0816_10380_1024.npy
robust_0819_0829_11880_1024.npy
robust_0819_0839_8200_512.npy
robust_0819_0911_32716_2048.npy
robust_0819_0929_16044_1024.npy
robust_0819_0942_11432_1024.npy
robust_0819_0955_11212_1024.npy
robust_0819_1012_7124_512.npy
robust_0819_1102_21168_1024.npy
robust_0819_1105_27528_2048.npy
robust_0819_1107_7276_512.npy
robust_0819_1109_21328_2048.npy
robust_0819_1116_51872_4096.npy
robust_0819_1119_14312_1024.npy
robust_0819_1120_5420_512.npy
robust_0819_1121_6856_512.npy
robust_0819_1124_17188_1024.npy
robust_0819_1130_5980_512.npy
robust_0819_1137_56520_4096.npy
robust_0819_1139_10444_1024.npy
robust_0819_1142_16200_1024.npy
robust_0819_1144_7484_512.npy
robust_0819_1148_12496_1024.npy
robust_0819_1152_11772_1024.npy
robust_0819_1153_11024_1024.npy
robust_0819_1155_12072_1024.npy
robust_0819_1158_15296_1024.npy
robust_0819_1203_29256_2048.npy
robust_0819_1209_28300_2048.npy
robust_0819_1213_16548_1024.npy]
'''
    for file_name in data_file_name:
        # load data
        history_load = np.load(file_name, allow_pickle=True)

        DATA_X = np.zeros([len(history_load), 1, 4, 4], dtype=np.float)
        DATA_Y = np.zeros([len(history_load), 1, 4], dtype=np.float)

        for i in range(len(history_load)):
            state = history_load[i]['state']
            action = history_load[i]['action']

            for j in range(4):
                for k in range(4):
                    DATA_X[i, 0, j, k] = state[j, k]

            DATA_Y[i, 0, action] = 1

        DATA_X = torch.from_numpy(DATA_X).float()
        DATA_Y = torch.from_numpy(DATA_Y).float()

            # print('state = ', state)
            # x_np = np.array(state, dtype=np.float)

            # x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).float()

            # print('x_np = ', x.type())

            # result = cnn.forward(x)
            # print(result.shape)
            # exit()

            # y = np.zeros([1, 4], dtype=np.float)
            # y[0, action] = 1

        for i in range(100):
            RES_Y = cnn.forward(DATA_X)

            loss = loss_function(RES_Y, DATA_Y)

            print('loss = ', loss.detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(cnn.state_dict(), 'test01.pkl')
