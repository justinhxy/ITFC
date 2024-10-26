from Net.basic import *
import torch.nn.functional as F

class MLPTableEncoder(nn.Module):
    def __init__(self, input_dim=9, output_dim=256):
        super(MLPTableEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

# 三模态融合模块
class TriModalCrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(TriModalCrossAttention, self).__init__()
        self.W_q1 = nn.Linear(input_dim, input_dim)
        self.W_k1 = nn.Linear(input_dim, input_dim)
        self.W_v1 = nn.Linear(input_dim, input_dim)

        self.W_q2 = nn.Linear(input_dim, input_dim)
        self.W_k2 = nn.Linear(input_dim, input_dim)
        self.W_v2 = nn.Linear(input_dim, input_dim)

        self.W_q3 = nn.Linear(input_dim, input_dim)
        self.W_k3 = nn.Linear(input_dim, input_dim)
        self.W_v3 = nn.Linear(input_dim, input_dim)

        self.W_o1 = nn.Linear(input_dim * 2, input_dim)
        self.W_o2 = nn.Linear(input_dim * 2, input_dim)
        self.W_o3 = nn.Linear(input_dim * 2, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x1, x2, x3):
        # x1, x2, x3: [B, N, input_dim] [8, 256, 1]
        batch_size, seq_len, _ = x1.size()

        # Linear transformations for each modality
        queries1 = self.W_q1(x1)
        keys2 = self.W_k2(x2)
        values2 = self.W_v2(x2)

        queries2 = self.W_q2(x2)
        keys3 = self.W_k3(x3)
        values3 = self.W_v3(x3)

        queries3 = self.W_q3(x3)
        keys1 = self.W_k1(x1)
        values1 = self.W_v1(x1)

        # Scaled dot-product attention
        attention_scores1 = torch.matmul(queries1, keys2.transpose(-2, -1)) / (x1.size(-1) ** 0.5)  # [B, N, N]
        attention_weights1 = F.softmax(attention_scores1, dim=-1)
        context1 = torch.matmul(self.dropout(attention_weights1), values2)  # [B, N, input_dim]

        attention_scores2 = torch.matmul(queries2, keys3.transpose(-2, -1)) / (x2.size(-1) ** 0.5)  # [B, N, N]
        attention_weights2 = F.softmax(attention_scores2, dim=-1)
        context2 = torch.matmul(self.dropout(attention_weights2), values3)  # [B, N, input_dim]

        attention_scores3 = torch.matmul(queries3, keys1.transpose(-2, -1)) / (x3.size(-1) ** 0.5)  # [B, N, N]
        attention_weights3 = F.softmax(attention_scores3, dim=-1)
        context3 = torch.matmul(self.dropout(attention_weights3), values1)  # [B, N, input_dim]
        # print('x1', x1.shape) # (2, 256, 1)
        # print('x2', x2.shape) # (2, 256, 1)
        # print('x3', x3.shape) # (2, 256, 1)
        # print('context1', context1.shape) # (2, 256, 1)
        # print('context2', context2.shape) # (2, 256, 1)
        # print('context3', context3.shape) # (2, 256, 1)


        # Concatenate context with input for each modality
        combined1 = torch.cat((x1, context1), dim=-1)  # [B, N, input_dim * 2]
        combined2 = torch.cat((x2, context2), dim=-1)  # [B, N, input_dim * 2]
        combined3 = torch.cat((x3, context3), dim=-1)  # [B, N, input_dim * 2]
        # print('combined1', combined1.shape) # (2, 256, 2)
        # print('combined2', combined2.shape) # (2, 256, 2)
        # print('combined3', combined3.shape) # (2, 256, 2)

        # Linear transformations and output for each modality
        output1 = self.W_o1(combined1)
        output2 = self.W_o2(combined2)
        output3 = self.W_o3(combined3)

        global_feature = torch.cat((output1, output2, output3), dim=1) # (2, 768, 1)
        return output1, output2, output3, global_feature

# class TriModalCrossAttention_ver2(nn.Module):
#     def __init__(self, input_dim):
#         super(TriModalCrossAttention_ver2, self).__init__()
#         self.W_q1 = nn.Linear(input_dim, input_dim)
#         self.W_k1 = nn.Linear(input_dim, input_dim)
#         self.W_v1 = nn.Linear(input_dim, input_dim)
#
#         self.W_q2 = nn.Linear(input_dim, input_dim)
#         self.W_k2 = nn.Linear(input_dim, input_dim)
#         self.W_v2 = nn.Linear(input_dim, input_dim)
#
#         self.W_q3 = nn.Linear(input_dim, input_dim)
#         self.W_k3 = nn.Linear(input_dim, input_dim)
#         self.W_v3 = nn.Linear(input_dim, input_dim)
#
#         self.W_o1 = nn.Linear(input_dim * 2, input_dim)
#         self.W_o2 = nn.Linear(input_dim * 2, input_dim)
#         self.W_o3 = nn.Linear(input_dim * 2, input_dim)
#         self.dropout = nn.Dropout(0.1)
#
#         self.q_linear = nn.Linear(input_dim * 3, input_dim)
#
#     def forward(self, x1, x2, x3):
#         # x1, x2, x3: [B, N, input_dim]
#         batch_size, seq_len, _ = x1.size()
#
#         # Linear transformations for each modality
#         queries1 = self.W_q1(x1)
#         keys2 = self.W_k2(x2)
#         values2 = self.W_v2(x2)
#
#         queries2 = self.W_q2(x2)
#         keys3 = self.W_k3(x3)
#         values3 = self.W_v3(x3)
#
#         queries3 = self.W_q3(x3)
#         keys1 = self.W_k1(x1)
#         values1 = self.W_v1(x1)
#
#         querie_all = torch.cat([queries1, queries2, queries3], dim=1)
#         querie_all = self.q_linear(querie_all)
#
#
#         # Scaled dot-product attention
#         attention_scores1 = torch.matmul(queries1, keys1.transpose(-2, -1)) / (x1.size(-1) ** 0.5)  # [B, N, N]
#         attention_weights1 = F.softmax(attention_scores1, dim=-1)
#         context1 = torch.matmul(self.dropout(attention_weights1), values2)  # [B, N, input_dim]
#
#         attention_scores2 = torch.matmul(queries2, keys2.transpose(-2, -1)) / (x2.size(-1) ** 0.5)  # [B, N, N]
#         attention_weights2 = F.softmax(attention_scores2, dim=-1)
#         context2 = torch.matmul(self.dropout(attention_weights2), values3)  # [B, N, input_dim]
#
#         attention_scores3 = torch.matmul(queries3, keys3.transpose(-2, -1)) / (x3.size(-1) ** 0.5)  # [B, N, N]
#         attention_weights3 = F.softmax(attention_scores3, dim=-1)
#         context3 = torch.matmul(self.dropout(attention_weights3), values1)  # [B, N, input_dim]
#
#         # Concatenate context with input for each modality
#         combined1 = torch.cat((x1, context1), dim=-1)  # [B, N, input_dim * 2]
#         combined2 = torch.cat((x2, context2), dim=-1)  # [B, N, input_dim * 2]
#         combined3 = torch.cat((x3, context3), dim=-1)  # [B, N, input_dim * 2]
#
#         # Linear transformations and output for each modality
#         output1 = self.W_o1(combined1)
#         output2 = self.W_o2(combined2)
#         output3 = self.W_o3(combined3)
#
#         global_feature = torch.cat((output1, output2, output3), dim=1) # [B, N * 3, input_dim ]
#         return output1, output2, output3, global_feature
#
#     def forward(self, x1, x2, x3):
#         # x1, x2, x3: [B, N, input_dim]
#         batch_size, seq_len, _ = x1.size()
#
#         # Linear transformations for each modality
#         queries1 = self.W_q1(x1)
#         keys1 = self.W_k1(x1)
#         values1 = self.W_v1(x1)
#
#         queries2 = self.W_q2(x2)
#         keys2 = self.W_k2(x2)
#         values2 = self.W_v2(x2)
#
#         queries3 = self.W_q3(x3)
#         keys3 = self.W_k3(x3)
#         values3 = self.W_v3(x3)
#
#         # 将三个模态的 Query 拼接在一起
#         query_all = torch.cat([queries1, queries2, queries3], dim=1)  # [B, 3 * N, input_dim] ,[B, N, input_dim]
#         # 分别计算三个模态的交叉注意力
#         attention_scores1 = torch.matmul(query_all, keys1.transpose(-2, -1)) / (x1.size(-1) ** 0.5)  # [B, 3 * N, N]
#         attention_weights1 = F.softmax(attention_scores1, dim=-1)
#         context1 = torch.matmul(self.dropout(attention_weights1), values1)  # [B, 3 * N, input_dim]
#
#         attention_scores2 = torch.matmul(query_all, keys2.transpose(-2, -1)) / (x2.size(-1) ** 0.5)  # [B, 3 * N, N]
#         attention_weights2 = F.softmax(attention_scores2, dim=-1)
#         context2 = torch.matmul(self.dropout(attention_weights2), values2)  # [B, 3 * N, input_dim]
#
#         attention_scores3 = torch.matmul(query_all, keys3.transpose(-2, -1)) / (x3.size(-1) ** 0.5)  # [B, 3 * N, N]
#         attention_weights3 = F.softmax(attention_scores3, dim=-1)
#         context3 = torch.matmul(self.dropout(attention_weights3), values3)  # [B, 3 * N, input_dim]
#
#         # Concatenate context with input for each modality
#         combined1 = torch.cat((x1, context1), dim=-1)  # [B, N, input_dim * 2]
#         combined2 = torch.cat((x2, context2), dim=-1)  # [B, N, input_dim * 2]
#         combined3 = torch.cat((x3, context3), dim=-1)  # [B, N, input_dim * 2]
#
#         # 对每个模态的输出进行线性变换
#         output1 = self.W_o1(context1)  # [B, 3 * N, input_dim]
#         output2 = self.W_o2(context2)  # [B, 3 * N, input_dim]
#         output3 = self.W_o3(context3)  # [B, 3 * N, input_dim]
#
#         # 拼接所有模态的输出
#         global_feature = torch.cat((output1, output2, output3), dim=1)  # [B, 3 * N, input_dim]
#
#         return output1, output2, output3, global_feature


class TriModalCrossAttention_ver2(nn.Module):
    def __init__(self, input_dim):
        super(TriModalCrossAttention_ver2, self).__init__()

        # 定义 Query, Key, Value 的线性变换
        self.W_q = nn.Linear(input_dim * 3, input_dim)  # 用于拼接后的查询
        self.W_k1 = nn.Linear(input_dim, input_dim)
        self.W_v1 = nn.Linear(input_dim, input_dim)

        self.W_k2 = nn.Linear(input_dim, input_dim)
        self.W_v2 = nn.Linear(input_dim, input_dim)

        self.W_k3 = nn.Linear(input_dim, input_dim)
        self.W_v3 = nn.Linear(input_dim, input_dim)

        # 输出投影
        self.W_o1 = nn.Linear(input_dim * 2, input_dim)
        self.W_o2 = nn.Linear(input_dim * 2, input_dim)
        self.W_o3 = nn.Linear(input_dim * 2, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x1, x2, x3):
        # x1, x2, x3: [B, N, input_dim] [8, 256, 1]
        batch_size, seq_len, input_dim = x1.size()

        # 将三个模态在最后一个维度拼接得到 [B, N, input_dim * 3]
        combined_input = torch.cat([x1, x2, x3], dim=-1)
        queries = self.W_q(combined_input)  # [B, N, input_dim * 3]

        # 对每个模态的键和值进行投影
        keys1 = self.W_k1(x1)  # [B, N, input_dim]
        values1 = self.W_v1(x1)

        keys2 = self.W_k2(x2)  # [B, N, input_dim]
        values2 = self.W_v2(x2)

        keys3 = self.W_k3(x3)  # [B, N, input_dim]
        values3 = self.W_v3(x3)

        # 分别计算每个模态的交叉注意力输出
        attention_scores1 = torch.matmul(queries, keys1.transpose(-2, -1)) / (input_dim ** 0.5)  # [B, N, N]
        attention_weights1 = F.softmax(attention_scores1, dim=-1)
        context1 = torch.matmul(self.dropout(attention_weights1), values1)  # [B, N, input_dim]

        attention_scores2 = torch.matmul(queries, keys2.transpose(-2, -1)) / (input_dim ** 0.5)  # [B, N, N]
        attention_weights2 = F.softmax(attention_scores2, dim=-1)
        context2 = torch.matmul(self.dropout(attention_weights2), values2)  # [B, N, input_dim]

        attention_scores3 = torch.matmul(queries, keys3.transpose(-2, -1)) / (input_dim ** 0.5)  # [B, N, N]
        attention_weights3 = F.softmax(attention_scores3, dim=-1)
        context3 = torch.matmul(self.dropout(attention_weights3), values3)  # [B, N, input_dim]

        # 将原输入与注意力上下文拼接
        combined1 = torch.cat((x1, context1), dim=-1)  # [B, N, input_dim * 2]
        combined2 = torch.cat((x2, context2), dim=-1)  # [B, N, input_dim * 2]
        combined3 = torch.cat((x3, context3), dim=-1)  # [B, N, input_dim * 2]

        # 线性变换生成最终输出
        output1 = self.W_o1(combined1)  # [B, N, input_dim]
        output2 = self.W_o2(combined2)  # [B, N, input_dim]
        output3 = self.W_o3(combined3)  # [B, N, input_dim]

        # 将三个模态的输出在第二维度拼接 [B, N * 3, input_dim]
        vision_feature = shuffle_interleave(output1, output2)
        # global_feature = torch.cat((output1, output2, output3), dim=1)
        global_feature = torch.cat((vision_feature,output3), dim=1)

        return output1, output2, output3, global_feature

class TriModalCrossAttention_ver2_2(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(TriModalCrossAttention_ver2_2, self).__init__()
        # 多头注意力机制
        self.cross_attention1 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention3 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # 用于将q拼接之后的维度映射到最终输出的维度
        self.q_linear = nn.Linear(input_dim * 3, input_dim)
        # self.linear = nn.Linear(3 * input_dim, output_dim)

    def forward(self, tensor1, tensor2, tensor3):
        # 假设输入的张量维度为 [B, N, input_dim]

        # 拼接三个模态张量作为q
        q = torch.cat((tensor1, tensor2, tensor3), dim=-1)  # [B, N, 3 * input_dim]
        q = self.q_linear(q)

        # 第一个交叉注意力：tensor1 作为 k, v
        attn_output1, _ = self.cross_attention1(q, tensor1, tensor1)  # [B, N, input_dim]

        # 第二个交叉注意力：tensor2 作为 k, v
        attn_output2, _ = self.cross_attention2(q, tensor2, tensor2)  # [B, N, input_dim]

        # 第三个交叉注意力：tensor3 作为 k, v
        attn_output3, _ = self.cross_attention3(q, tensor3, tensor3)  # [B, N, input_dim]

        # 将三个交叉注意力的输出拼接
        # combined_output = torch.cat((attn_output1, attn_output2, attn_output3), dim=2)  # [B, N, 3 * input_dim]
        combined_output = torch.cat((attn_output1, attn_output2, attn_output3), dim=1)  # [B, 3 * N, input_dim]

        # 将拼接后的输出映射到最终的输出维度
        # output = self.linear(combined_output)  # [B, N, output_dim]

        return combined_output

class Triple_model_CrossAttentionFusion(nn.Module):
    def __init__(self):
        super(Triple_model_CrossAttentionFusion, self).__init__()
        self.name = 'Triple_model_CrossAttentionFusion'
        self.Resnet = get_pretrained_Vision_Encoder() # input [B,C,128,128,128] OUT[8.400]
        self.Table_linear = MLPTableEncoder()
        self.fc_vis = nn.Linear(400, 256)
        self.fusion = TriModalCrossAttention(input_dim=1)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=2)

    def forward(self, mri, pet, cli):
        mri_feature = self.Resnet(mri)
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.Resnet(pet)
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table_linear(cli)
        # print(f'cli feature shape:{cli_feature.shape}')
        mri_feature = torch.unsqueeze(mri_feature, dim=-1)
        pet_feature = torch.unsqueeze(pet_feature, dim=-1)
        cli_feature = torch.unsqueeze(cli_feature, dim=-1)
        mri_feature, pet_feature, cli_feature, global_feature = self.fusion(mri_feature, pet_feature, cli_feature)
        global_feature = global_feature[-1].permute(0, 2, 1)
        output = self.classify_head(global_feature)
        return output

class Triple_model_CrossAttentionFusion_KAN(nn.Module):
    def __init__(self):
        super(Triple_model_CrossAttentionFusion_KAN, self).__init__()
        self.name = 'Triple_model_CrossAttentionFusion_KAN'
        self.Resnet = get_pretrained_Vision_Encoder() # input [B,C,128,128,128] OUT[8.400]
        self.Table = TransformerEncoder()
        self.fc_vis = nn.Linear(400, 256)
        self.fusion = TriModalCrossAttention(input_dim=1)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=2)

    def forward(self, mri, pet, cli):
        mri_feature = self.Resnet(mri)
        pet_feature = self.Resnet(pet)
        cli_feature = self.Table(cli)
        mri_feature = torch.unsqueeze(mri_feature, dim=-1)
        pet_feature = torch.unsqueeze(pet_feature, dim=-1)
        cli_feature = torch.unsqueeze(cli_feature, dim=-1)
        mri_feature, pet_feature, cli_feature, global_feature = self.fusion(mri_feature, pet_feature, cli_feature)
        global_feature = global_feature[-1].permute(0, 2, 1)
        output = self.classify_head(global_feature)
        return mri_feature, pet_feature, cli_feature, output

class Triple_model_CrossAttentionFusion_self(nn.Module):
    def __init__(self):
        super(Triple_model_CrossAttentionFusion_self, self).__init__()
        self.name = 'Triple_model_CrossAttentionFusion_self'
        self.Resnet = get_pretrained_Vision_Encoder() # input [B,C,128,128,128] OUT[8.400]
        # self.Table = TransformerEncoder(output_dim=256)
        self.Table = MLPTableEncoder()
        self.fc_vis = nn.Linear(400, 256)
        self.fusion = TriModalCrossAttention(input_dim=1)
        self.SA1 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA2 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA3 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=2)

    def forward(self, mri, pet, cli):
        mri_feature = self.Resnet(mri)
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.Resnet(pet)
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table(cli)

        mri_feature = self.fc_vis(mri_feature)
        pet_feature = self.fc_vis(pet_feature)
        # cli_feature = self.fc_cli(cli_feature)

        mri_feature = torch.unsqueeze(mri_feature, dim=1)
        pet_feature = torch.unsqueeze(pet_feature, dim=1)
        cli_feature = torch.unsqueeze(cli_feature, dim=1)

        mri_feature = self.SA2(mri_feature)
        pet_feature = self.SA3(pet_feature)
        cli_feature = self.SA1(cli_feature)

        mri_feature_tr = mri_feature.permute(0, 2, 1)
        pet_feature_tr = pet_feature.permute(0, 2, 1)
        cli_feature_tr = cli_feature.permute(0, 2, 1)
        _, _, _, global_feature  = self.fusion(mri_feature_tr, pet_feature_tr, cli_feature_tr)
        global_feature = global_feature.permute(0, 2, 1)
        output = self.classify_head(global_feature)
        return mri_feature, pet_feature, cli_feature, output

class Triple_model_CrossAttentionFusion_self_KAN(nn.Module):
    def __init__(self):
        super(Triple_model_CrossAttentionFusion_self_KAN, self).__init__()
        self.name = 'Triple_model_CrossAttentionFusion_self_KAN'
        self.Resnet = get_pretrained_Vision_Encoder() # input [B,C,128,128,128] OUT[8.400]
        self.Table = TransformerEncoder(output_dim=256)
        self.fc_vis = nn.Linear(400, 256)

        # self.fusion = TriModalCrossAttention(input_dim=1)
        self.fusion = TriModalCrossAttention_ver2(input_dim=1)
        # self.fusion = TriModalCrossAttention_ver2_2(input_dim=1, output_dim=1, num_heads=1)

        self.SA1 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA2 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.SA3 = SelfAttention(16, 256, 256, hidden_dropout_prob=0.2)
        self.classify_head = DenseNet(layer_num=(6, 12, 24, 16), growth_rate=16, in_channels=1, classes=2)

    def forward(self, mri, pet, cli):
        mri_feature = self.Resnet(mri)
        # print(f'mri feature shape: {mri_feature.shape}')
        pet_feature = self.Resnet(pet)
        # print(f'pet feature.shape:{pet_feature.shape}')
        cli_feature = self.Table(cli)

        mri_feature = self.fc_vis(mri_feature)
        pet_feature = self.fc_vis(pet_feature)
        # cli_feature = self.fc_cli(cli_feature)

        mri_feature = torch.unsqueeze(mri_feature, dim=1)
        pet_feature = torch.unsqueeze(pet_feature, dim=1)
        cli_feature = torch.unsqueeze(cli_feature, dim=1)

        mri_feature = self.SA2(mri_feature)
        pet_feature = self.SA3(pet_feature)
        cli_feature = self.SA1(cli_feature)
        # mri_feature.shape torch.Size([8, 1, 256])
        mri_feature_tr = mri_feature.permute(0, 2, 1)
        pet_feature_tr = pet_feature.permute(0, 2, 1)
        cli_feature_tr = cli_feature.permute(0, 2, 1)

        # print('mri_feature_tr', mri_feature_tr.shape) # torch.Size([8, 256, 1])
        # print('pet_feature_tr', pet_feature_tr.shape) # torch.Size([8, 256, 1])
        # print('cli_feature_tr', cli_feature_tr.shape) # torch.Size([8, 256, 1])

        # _, _, _, global_feature  = self.fusion(mri_feature_tr, pet_feature_tr, cli_feature_tr)
        global_feature  = self.fusion(mri_feature_tr, pet_feature_tr, cli_feature_tr)
        # print('global_feature', global_feature.shape) # (2, 768, 1)

        global_feature = global_feature.permute(0, 2, 1)  # (2, 1, 768)
        output = self.classify_head(global_feature) # (2, 2)
        return mri_feature, pet_feature, cli_feature, output