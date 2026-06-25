import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

class BilinearFusion(nn.Module):
    def __init__(self, dim1, dim2, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_dim, dim1, dim2))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, y):
        # x: (B, dim1), y: (B, dim2)
        batch_size = x.size(0)
        output = torch.einsum('bi,ojk,bj->bok', x, self.weight, y)
        output = output.view(batch_size, -1) + self.bias
        return output

class AttentionFusion(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, out_dim):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, out_dim)
        self.key_proj = nn.Linear(key_dim, out_dim)
        self.value_proj = nn.Linear(value_dim, out_dim)
        self.attention = nn.MultiheadAttention(out_dim, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, query, key, value):
        Q = self.query_proj(query).unsqueeze(1) # (B, 1, out_dim)
        K = self.key_proj(key).unsqueeze(1)     # (B, 1, out_dim)
        V = self.value_proj(value).unsqueeze(1)   # (B, 1, out_dim)
        attn_output, _ = self.attention(Q, K, V) # (B, 1, out_dim)
        return self.norm(attn_output.squeeze(1))  # (B, out_dim)

@DETECTOR.register_module(module_name='selfi')
class SELFIDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.dropout = config['backbone_config'].get('dropout', False)
        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        # Conditions
        self.IAFM = config['backbone_config'].get('IAFM', False)
        self.fusion_method = config['backbone_config'].get('fusion_method', False)
        self.relevance_loss = config['backbone_config'].get('relevance_loss', False)
        self.cosine_diversity = config['backbone_config'].get('cosine_diversity', False)
        self.relevance_predictor_type = config['backbone_config'].get('relevance_predictor', False)
        self.use_identity_real_fake_head = config['backbone_config'].get('use_identity_real_fake_head', False)

        if config['backbone_name'] == 'efficientnetb4':
            self.feat_dim = 1792
            self.id_feat_dim = 512
        elif config['backbone_name'] == 'clip':
            self.feat_dim = 768
            self.id_feat_dim = 512
        elif config['backbone_name'] == 'xception':
            self.feat_dim = 2048
            self.id_feat_dim = 512
        elif config['backbone_name'] == 'resnet34':
            self.feat_dim = 512
            self.id_feat_dim = 512    
        else:
            raise ValueError(f"Unsupported backbone name: {config['backbone_name']}")

        self.real_or_fake_head = nn.Linear(self.feat_dim, 2)  # Default output dim
    
        if self.IAFM:
            self.iresnet = self.build_face_recognition_model(config)
            self.proj_id = nn.Linear(self.id_feat_dim, self.feat_dim, bias=False)
            self.proj_id_norm = nn.LayerNorm(self.feat_dim)
            
            if self.use_identity_real_fake_head:
                self.identity_real_fake_head = nn.Linear(self.feat_dim, 2)

            self.feat_norm = nn.LayerNorm(self.feat_dim)  # Feature normalization

            if self.fusion_method == 'weighted_sum':
                if self.relevance_predictor_type == 'concat':
                    self.relevance_predictor = nn.Sequential(
                        nn.Linear(self.feat_dim*2, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1),
                        nn.Sigmoid()
                    )
                elif self.relevance_predictor_type in ['identity_mlp', 'visual_mlp', 'diff', 'product']:
                    self.relevance_predictor = nn.Sequential(
                        nn.Linear(self.id_feat_dim, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1),
                        nn.Sigmoid()
                    )
                elif self.relevance_predictor_type == 'dot':
                    self.relevance_predictor = None
                else:
                    raise ValueError(f"Unsupported relevance predictor type: {self.relevance_predictor_type}")
            elif self.fusion_method == 'concat':
                self.real_or_fake_head = nn.Linear(self.feat_dim*2, 2)
            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
            
            
        self.cls_loss_func = self.build_loss(config['loss_func']['cls_loss'])
        if self.IAFM and self.cosine_diversity:
            self.cosine_loss_func = self.build_loss(config['loss_func']['cosine_diversity_loss'])
        if self.IAFM and self.relevance_loss and self.fusion_method == 'weighted_sum':
            self.relevance_loss_func = self.build_loss(config['loss_func']['relevance_loss'])

        self.cls_weight = config['loss_weight'].get('cls_weight', 1)
        self.relevance_weight = config['loss_weight'].get('relevance_weight', 0.5)
        self.identity_cls_weight = config['loss_weight'].get('identity_cls_weight', 1)
        self.cosine_diversity_weight = config['loss_weight'].get('cosine_diversity_weight', 1)
        

        logger.info(f"IAFM: {self.IAFM}, Fusion Method: {self.fusion_method}, Relevance Loss: {self.relevance_loss}, Fusion Method: {self.fusion_method},Relevance Predictor: {self.relevance_predictor_type}")
    
    def build_backbone(self, config):
        if config['backbone_name'] == 'clip':
            _, backbone = get_clip_visual(model_name="openai/clip-vit-base-patch16")
            logger.info('CLIP - Load pretrained model successfully!')
        elif config['backbone_name'] == 'efficientnetb4':
            backbone_class = BACKBONE[config['backbone_name']]
            model_config = config['backbone_config']
            model_config['pretrained'] = config['pretrained']
            backbone = backbone_class(model_config)
            logger.info('EfficientNetB4 - Load pretrained model successfully!' if config['pretrained'] else 'EfficientNetB4 - No pretrained model.')
        elif config['backbone_name'] == 'xception':
            backbone_class = BACKBONE[config['backbone_name']]
            model_config = config['backbone_config']
            backbone = backbone_class(model_config)
            if self.config['pretrained']:
                state_dict = torch.load(self.config['pretrained'])
                for name, weights in state_dict.items():
                    if 'pointwise' in name:
                        state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
                        
                state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
                backbone.load_state_dict(state_dict, False)
                logger.info('Xception - Load pretrained model successfully!')
            else:
                logger.info('Xception - No pretrained model.')
        elif config['backbone_name'] == 'resnet34':
            backbone_class = BACKBONE[config['backbone_name']]
            model_config = config['backbone_config']
            backbone = backbone_class(model_config)
        else:
            raise ValueError(f"Unsupported backbone name: {config['backbone_name']}")
        return backbone

    def build_loss(self, name, tau=None):
        loss_class = LOSSFUNC[name]
        return loss_class(tau) if tau else loss_class()

    def build_face_recognition_model(self, config):
        backbone_class = BACKBONE[config['face_model_name']]
        pretrained_path = config.get('face_model_pretrained', None)
        if not pretrained_path:
            raise ValueError("Pretrained weights for IResNet not found in config['pretrained_path'].")

        iresnet = backbone_class(pretrained_path)
        for param in iresnet.parameters():
            param.requires_grad = False
        return iresnet

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        real_fake_loss = self.cls_loss_func(pred_dict['cls'], data_dict['label'])
        relevance_loss = torch.tensor(0)
        identity_real_fake_loss = torch.tensor(0)
        cosine_loss = torch.tensor(0)
        if self.IAFM:
            if self.use_identity_real_fake_head:
                identity_real_fake_loss = self.cls_loss_func(pred_dict['identity_cls'], data_dict['label'])
            
            if self.cosine_diversity:
                cosine_loss = self.cosine_loss_func(
                    input1=pred_dict['LN_identity_feat'],
                    input2=pred_dict['LN_feat']
                )
            if self.relevance_loss and self.fusion_method == 'weighted_sum':
                relevance_loss = self.relevance_loss_func(
                    fused_feat=pred_dict['fused_feat'],
                    id_feat=pred_dict['identity_feat'],
                    cnn_feat=pred_dict['feat'],
                    relevance=pred_dict['relevance']
                )
        loss = (self.cls_weight * real_fake_loss) + (self.identity_cls_weight * identity_real_fake_loss) + (self.relevance_weight * relevance_loss) + (self.cosine_diversity_weight * cosine_loss)
        return {
            'overall': loss,
            'real_or_fake': real_fake_loss,
            'identity_real_or_fake': identity_real_fake_loss,
            'cosine_loss': cosine_loss,
            'relevance_loss': relevance_loss,
            'avg_relevance': torch.mean(pred_dict['relevance']) if pred_dict.get('relevance') is not None else torch.tensor(0)
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label, pred = data_dict['label'], pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def features(self, data_dict: dict) -> torch.tensor:
        if self.config['backbone_name'] == 'clip':
            x = self.backbone(data_dict['image'])['pooler_output']
        elif self.config['backbone_name'] == 'efficientnetb4':
            x = self.backbone.features(data_dict['image'])
            if x.size(1) >= 2:
                x = F.adaptive_avg_pool2d(x, (1,1))
                x = x.view(x.size(0), -1)
                
                if self.dropout:
                    x = self.dropout_layer(x)
        elif self.config['backbone_name'] == 'xception':
            x = self.backbone.features(data_dict['image'])
            if len(x.shape) == 4:
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
        elif self.config['backbone_name'] == 'resnet34':
            x = self.backbone.features(data_dict['image'])
            if len(x.shape) == 4:
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
        return x

    def get_face_embedding(self, x) -> torch.tensor:
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        with torch.no_grad():
            embedding = self.iresnet.features(x)
            projected = self.proj_id(embedding)
            LN_projected = self.proj_id_norm(projected)
        return projected, LN_projected

    def get_relevance(self, projected_face_features: torch.tensor, features: torch.tensor) -> torch.tensor:
        if self.relevance_predictor_type == 'concat':
            x = torch.cat([projected_face_features, features], dim=1)
        elif self.relevance_predictor_type == 'identity_mlp':
            x = projected_face_features
        elif self.relevance_predictor_type == 'visual_mlp':
            x = features
        else:
            raise ValueError(f"Unsupported relevance predictor type: {self.relevance_predictor_type}")
        return self.relevance_predictor(x)

    def get_fused_features(self, projected_face_features: torch.tensor, features: torch.tensor, relevance=None) -> torch.tensor:
        if self.fusion_method == 'weighted_sum' or self.fusion_method == 'weighted_sum_feature':
            return (1 - relevance) * features + relevance * projected_face_features
        elif self.fusion_method == 'concat':
            return torch.cat([features, projected_face_features], dim=1)
        elif self.fusion_method == 'bilinear':
            return self.fusion_layer(features, projected_face_features)
        elif self.fusion_method == 'attention':
            return self.attention_layer(features, projected_face_features, projected_face_features)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.real_or_fake_head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)

        relevance = None
        projected_face_features = None
        fused_features = None
        identity_cls = None
        LN_features = None
        LN_projected_face_features = None

        if self.IAFM:
            projected_face_features, LN_projected_face_features = self.get_face_embedding(data_dict['image'])
            LN_features = self.feat_norm(features)  # Apply LayerNorm here
            
            if self.use_identity_real_fake_head:
                identity_cls = self.identity_real_fake_head(projected_face_features)

            if self.fusion_method == 'weighted_sum' or self.fusion_method == 'weighted_sum_feature':
                relevance = self.get_relevance(LN_projected_face_features, LN_features)
                fused_features = self.get_fused_features(LN_projected_face_features, LN_features, relevance)
            elif self.fusion_method == 'concat':
                fused_features = self.get_fused_features(LN_projected_face_features, LN_features)
            else:
                fused_features = self.get_fused_features(LN_projected_face_features, LN_features)

            cls = self.classifier(fused_features)
        else:
            cls = self.classifier(features)

        prob = torch.softmax(cls, dim=1)[:, 1]

        return {
            'cls': cls,
            'identity_cls': identity_cls,
            'prob': prob,
            'feat': features,
            'identity_feat': projected_face_features,
            'LN_feat': LN_features,
            'LN_identity_feat': LN_projected_face_features,
            'fused_feat': fused_features,
            'relevance': relevance,
        }
        
        
def get_clip_visual(model_name = "openai/clip-vit-base-patch16"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model
