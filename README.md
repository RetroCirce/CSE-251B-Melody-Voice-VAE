# CSE-251B-Melody-Voice-VAE
To learn the representation of polyphonic music 

model文件夹下包含了模型的核心文件，不过目前这个模型存在一系列小问题，同时这个模型是没有dynamic padding的，我们后期要修正

train和eval文件就是训练和测试的代码了，现在虽然progress report中说我们是用test集测得，但是实际上我们现在还是用val集合（

recon_demo 和 result下保存了一些结果和可视化的组件，大部分只要由@Ke来读取和运行就可以了

loader文件夹下保存了处理数据，分层旋律的一些代码，这个是没有dynamic padding的，我们后期要修正

merge和midi_decode是负责把输出的monophonic整合成polyphonic同时输出为可以听的midi

我们接下来主要要修改的地方有：

1. model文件夹下的模型文件，加入dynamic padding，修正一些小错误
2. 修改训练和测试代码，加入dynamic padding，同时优化一些超参
3. 希望能够处理2小节的音乐，而不是1小节的，从目前来看一小节的旋律插值效果不是很酷
4. 对比pianotree，这个基本上由@Ke来运行，代码和模型都已经有了
5. loss等图的输出

关于四种数据的说明：训练/验证/测试数据大概比例为 29000:3500:3500 （最后决定不用overlap）

1. fix:定长补足10小节的数据，大小为 [N, 320]
2. dynamic: 不定长减少冗余，大小为 [N, 不定长]
3. dynamic_sub： 不定长+子图划分，划分依据是每个小节的最高最低音
4. dynamic_sub_pr: 不定长+子图划分，划分依据是固定的 [24, 48, 60, 108]
