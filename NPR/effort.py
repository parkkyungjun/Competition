import torch
import torch.nn as nn
import timm
import math

class EffortLinear(nn.Module):
    """
    [Paper Implementation]
    SVD-based Orthogonal Decomposition Layer.
    W = W_semantic (Frozen) + W_forgery (Trainable)
    """
    def __init__(self, original_linear: nn.Linear, rank_ratio: float = 0.8):
        super().__init__()
        
        # 1. 원본 가중치 및 바이어스 추출
        weight = original_linear.weight.data # (Out, In)
        out_features, in_features = weight.shape
        
        # Bias 처리
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data)
        else:
            self.register_parameter('bias', None)

        # 2. SVD 수행 (Full Accuracy를 위해 float32 변환 후 수행 권장)
        # W = U @ S @ Vh
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        
        # 3. Rank k 결정 (Energy Preserving Ratio)
        full_rank = min(out_features, in_features)
        k = int(full_rank * rank_ratio)
        if k == 0: k = 1 # 최소 1개는 보장
        
        # 4. [Semantic Subspace] - Frozen (학습 X)
        # 상위 k개의 고유값 성분은 일반화된 특징으로 간주하여 고정
        self.register_buffer('U_p', U[:, :k].clone())
        self.register_buffer('S_p', S[:k].clone())
        self.register_buffer('Vh_p', Vh[:k, :].clone())
        
        # 5. [Forgery Subspace] - Trainable (학습 O)
        # 하위 성분(Residual)을 초기값으로 하여 미세 조정(Fine-tuning)
        # Deepfake의 미세한 아티팩트는 이 잔차 공간에서 학습됨
        self.U_r = nn.Parameter(U[:, k:].clone())
        self.S_r = nn.Parameter(S[k:].clone())
        self.Vh_r = nn.Parameter(Vh[k:, :].clone())

    def forward(self, x):
        # Semantic Path (Fixed)
        W_semantic = self.U_p @ torch.diag(self.S_p) @ self.Vh_p
        
        # Forgery Path (Trainable)
        W_trainable = self.U_r @ torch.diag(self.S_r) @ self.Vh_r
        
        # Reconstruct Weight
        W_total = W_semantic + W_trainable
        
        # 원래 데이터 타입(fp16/bf16 등)으로 캐스팅하여 연산
        return nn.functional.linear(x, W_total.type_as(x), self.bias)

def inject_effort_layers(model, rank_ratio=0.75, target_modules=['head']):
    """
    모델 내부를 순회하며 지정된 이름(target_modules)을 포함하는 Linear 레이어를
    EffortLinear로 교체합니다.
    
    Args:
        target_modules: Swin의 경우 ['head', 'fc1', 'fc2', 'qkv', 'proj'] 등이 타겟
    """
    for name, module in model.named_children():
        # 현재 모듈이 타겟 모듈 이름을 포함하고, Linear 레이어인 경우 교체
        if isinstance(module, nn.Linear):
            # 타겟 모듈 리스트 중 하나라도 이름에 포함되는지 확인
            is_target = any(t in name for t in target_modules)
            # 혹은 부모 모듈의 이름으로 판단해야 할 경우(보통 재귀 호출 안에서 처리됨)
            # 여기서는 단순화를 위해 모든 Linear를 타겟팅하거나, 특정 이름만 타겟팅
            
            # Swin Transformer(timm) 구조상 이름 매칭이 중요
            # 보통 재귀함수 밖에서 모듈 이름을 체크하는 것이 안전하나, 
            # 여기서는 편의상 모듈 교체 로직을 수행
            pass 

    # 재귀적으로 모듈 교체 수행
    replace_modules(model, rank_ratio, target_modules)
    return model

def replace_modules(model, rank_ratio, target_suffixes):
    """
    재귀적으로 모델 트리를 내려가며 레이어를 교체 (In-place modification)
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            # 현재 레이어 이름이 타겟 접미사 중 하나로 끝나는지 확인
            # 예: block.0.attn.qkv, block.0.mlp.fc1 ...
            if any(name.endswith(suffix) for suffix in target_suffixes):
                print(f"Applying Effort to: {name}")
                new_layer = EffortLinear(child, rank_ratio=rank_ratio)
                setattr(model, name, new_layer)
        else:
            # 자식 모듈로 재귀 진입
            replace_modules(child, rank_ratio, target_suffixes)

class DeepfakeEffortModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window12_384', num_classes=2, rank_ratio=0.75):
        super().__init__()
        
        print(f"Loading Backbone: {model_name}...")
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # 논문/대회 세팅: 일반적으로 Attention의 Projection과 MLP를 모두 튜닝함
        # timm swin transformer 구현체 기준 타겟 모듈명:
        # - Attention: 'qkv', 'proj'
        # - MLP: 'fc1', 'fc2'
        # - Classifier: 'head'
        target_layers = ['qkv', 'proj', 'fc1', 'fc2', 'head']
        
        print(f"Injecting Effort Layers (Rank Ratio: {rank_ratio})...")
        print(f"Target Modules: {target_layers}")
        
        # 1. 레이어 교체 (In-place)
        replace_modules(self.backbone, rank_ratio, target_layers)
        
        # 2. Gradient 설정 (Frozen vs Trainable)
        # 기본적으로 Backbone의 파라미터는 Freeze하고, 
        # EffortLinear 내부의 U_r, S_r, Vh_r만 requires_grad=True가 됨(자동 설정됨)
        # 하지만 LayerNorm 등 다른 파라미터는 상황에 따라 켜고 끌 필요가 있음.
        
        # 일단 전체 Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # EffortLayer의 Trainable 파트만 Unfreeze
        trainable_count = 0
        for name, module in self.backbone.named_modules():
            if isinstance(module, EffortLinear):
                module.U_r.requires_grad = True
                module.S_r.requires_grad = True
                module.Vh_r.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True
                
                trainable_count += module.U_r.numel() + module.S_r.numel() + module.Vh_r.numel()

        print(f"Model Ready. Trainable Parameters (SVD Residuals): {trainable_count:,}")

    def forward(self, x):
        return self.backbone(x)