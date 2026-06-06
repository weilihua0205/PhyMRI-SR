import random
import math
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from datasets import register
from utils import to_pixel_samples


def _compute_foreground_mask(arr, fg_threshold=None):
    """
    Compute a foreground mask for array `arr` (numpy) with shape [C,H,W] or [H,W].
    Returns mask with same shape as arr (broadcastable) where foreground pixels are 1.0.
    If fg_threshold is None, use (min+max)/2 on the first channel.
    """
    a = arr
    if a.ndim == 3:
        img = a[0]
    else:
        img = a
    if fg_threshold is None:
        tmin = float(np.min(img))
        tmax = float(np.max(img))
        threshold = (tmin + tmax) / 2.0
    else:
        threshold = float(fg_threshold)
    mask2d = (img > threshold).astype(np.float32)
    # expand to channels
    if a.ndim == 3:
        mask = np.repeat(mask2d[np.newaxis, ...], a.shape[0], axis=0)
    else:
        mask = mask2d
    return mask


def _add_rician_noise_to_magnitude(img, sigma, k_factor=1.0, mode='rician', foreground_only=False, fg_threshold=None):
    """
    缂傚倸鍊搁崐鐑芥倿閿曞倹鍋￠柕澶堝劤閺嗭箓鏌ｉ弬鍨倯闁绘帞绮幈銊ノ熺紒妯荤€繝銏ｎ潐濞叉﹢銆冮妷鈺傚€烽柛娆忣槸濞咃綁鏌ｆ惔銈庢綈濠电偛锕ら悾鐑藉醇閺囩喎鈧攱銇勯幒鍡椾壕濠电偛鍚嬮崹鍧楀蓟?MRI 婵犵绱曢崑娑㈩敄閸涱垪鍋撳☉鎺撴珚闁诡啫鍥х濞达絿鎳撻崜顓㈡⒑閸涘﹥澶勯柛顭戜邯瀹曟垿骞樼拠鑼姦濡炪倖甯掔€氼剛绮堝畝鍕厸鐎广儱娲﹂弳鈺冪磼閹邦厹鈧梻绱撻崒娆戭槮闁哄牜鍓欓—鍐寠婢光晜鐩畷绋课旈埀顒傜不閿濆棎浜滈煫鍥ㄦ尰閸ｆ椽鏌?+ Rician 闂傚倷绶氬鑽ょ礊閸℃顩叉繝濠傜墛閺咁剟鏌熼悧鍫熺凡鏉?    - k_factor: K缂傚倸鍊风粈渚€寮甸鈧—鍐寠婢光晜鐩獮鎰償閿濆倹鏁垫繝鐢靛█濞佳囨偋婵犲倵鏋旈悘鐐靛亾閸欏繘鏌ㄥ┑鍡橆棡闁绘搫绲跨槐?(0-1)闂?1闂傚倷绀侀幖顐﹀疮椤栫偛纾块柤娴嬫櫆瀹曞弶淇婇姘倯闁稿锕㈤幃姗€鎮欑捄杞版睏缂備礁顦遍弫濠氬蓟濞戙垹绠抽柟瀵稿剱濡儲淇婇悙鑼憼闁诡喖鍊块獮鍐箮閽樺鍊炲銈庡幖婵傛梻娆㈠鍓佸祦?    - sigma: Rician闂傚倷绶氬鑽ょ礊閸℃顩叉繝濠傜墛閺咁剟鏌熼幍顔碱暭闁稿骸绉归弻娑㈠即閵娿儱顫梺鎼炲妼閵堟悂寮诲☉銏犵婵犻潧娴傚Λ鐐测攽?    - mode: 'rician' (婵犳鍠楃敮妤冪矙閹烘せ鈧箓宕奸妷顔芥櫍? 闂?'gaussian' (婵犵數鍋涢顓熸叏閹绢喖绠犻幖娣灪閸欏繘鏌ｉ幇顓熷剹婵炲牊顨嗘穱濠囶敍濠婂懎绗￠梺缁樺笒濞硷繝寮诲☉銏℃櫜閹煎瓨绻勯惄搴ㄦ⒑?

    闂備礁鎼ˇ顖炴偋婵犲洤绠伴柟闂寸閻?闂備礁鎼ˇ顖炴偋婵犲洤绠伴柟闂寸閸氳銇勯幘鍗炵仼闁告劏鍋撻梻浣虹帛閹哥霉妞嬪孩鍏滈柍褜鍓熷娲箰鎼达絺濮囬梺娲诲弾閸犳岸骞戦姀鐙€娼╅悹楦挎椤?torch Tensor闂傚倷鐒︾€笛呯矙閹存繐鑰块柟杈ㄦ懕闂傚倷鐒︾€笛呯矙閹次诲洦瀵奸弶鎴濈€?numpy ndarray闂傚倷鐒︾€笛呯矙閹存繐鑰块柟杈ㄦ懕闂傚倷鐒︾€笛呯矙閹次诲洦娼忛埡鍌氫粡濡炪倖鍔х粻鎴﹀垂閸岀偞鍊甸柨婵嗗€瑰▍鍥ㄣ亜韫囨稐鎲鹃柡?    """
    # 婵犵數濮伴崹鐓庘枖濞戞埃鍋撳鐓庢珝妤?numpy
    if isinstance(img, np.ndarray):
        arr = img.copy()
        # 婵犵數鍎戠徊钘壝洪敂鐐床闁糕剝绋掗崕濠傗攽婢跺顑圵
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        # 缂傚倷鑳堕搹搴ㄥ矗鎼淬劌绐楅柡鍥╁У瀹曞弶鎱ㄥ┑鍫濈€給at32
        arr = arr.astype(np.float32)
        
        if mode == 'gaussian':
            result = arr + np.random.normal(0, sigma, size=arr.shape)
            if foreground_only:
                # apply mask so background remains original
                mask = _compute_foreground_mask(arr, fg_threshold)
                result = mask * result + (1.0 - mask) * arr
            return result
        
        # rician with kspace degradation
        degraded = _apply_kspace_degradation_and_rician_noise(arr, sigma, k_factor, foreground_only, fg_threshold)
        return degraded

    # 婵犵數濮伴崹鐓庘枖濞戞埃鍋撳鐓庢珝妤?torch Tensor
    if isinstance(img, torch.Tensor):
        t = img.clone().to(dtype=torch.float32)
        
        if mode == 'gaussian':
            result = t + torch.randn_like(t) * float(sigma)
            if foreground_only:
                np_orig = t.detach().cpu().numpy()
                np_result = result.detach().cpu().numpy()
                mask = _compute_foreground_mask(np_orig, fg_threshold)
                np_masked = mask * np_result + (1.0 - mask) * np_orig
                return torch.from_numpy(np_masked).to(t.device, dtype=t.dtype)
            return result
        
        # rician with kspace degradation
        np_arr = t.detach().cpu().numpy()
        degraded_np = _apply_kspace_degradation_and_rician_noise(np_arr, sigma, k_factor, foreground_only, fg_threshold)
        return torch.from_numpy(degraded_np).to(t.device, dtype=t.dtype)

    # Return the original input unchanged for unsupported types.
    return img

def _apply_kspace_degradation_and_rician_noise(arr, sigma, k_factor, foreground_only=False, fg_threshold=None):
    """
    闂傚倷绀侀幉锟犲礉閺囥垹绠犳慨妞诲亾鐎规洘娲熼獮姗€顢欓懖鈺婃П闂備礁鍚嬫禍浠嬪磿閹绘巻鏋旈柛顐ｆ礃閻撱儲绻涢幋鐐垫噯濠㈣蓱閵囧嫰寮撮悩铏彅umpy闂傚倷娴囧銊╂倿閿旂晫鐝堕柛鈩冪懃閸?(CHW) 闂備礁婀遍崢褔鎮洪妸銉綎濠电姵鑹鹃弸浣广亜閵夈劋鍚紒杈ㄥ笧娴狅箓鎮欓鍙ヨ檸婵犵妲呴崑鍕焽閿熺姷宓侀悗锝庡櫘閺佸倿鏌涘☉鍗炴灓闁告梻鍏樺娲川婵犲嫮鐓€濡炪倖宸婚弳鎭坕an闂傚倷绶氬鑽ょ礊閸℃顩叉繝濠傜墛閺咁剟鏌熼悧鍫熺凡鏉?    闂傚倷鑳堕…鍫ヮ敄閸涱劶娲煛閸滀焦鏅╅梺?1闂傚倷鐒︾€笛呯矙閹达附鍋嬮柛鈩冾殘閺嗭箓鏌ｉ弬鍨倯闁绘帞绮幈銊ノ熺紒妯荤€繝銏ｎ潐濡插嚧I闂傚倷鐒︾€笛呯矙閹次诲洭顢橀姀鐘靛姦?    """
    if arr.shape[0] != 1:
        # 婵犵數濮伴崹濂稿春閺嵮呮殕闁归棿绀侀悞鍨亜閹哄秶顦﹂柛婵嗘惈闇夋繝濠傜墢閻ｆ椽鏌℃担鍝バ㈡い顐ｇ箞椤㈡寰勬繝鍐惧晠濠电姵顔栭崳顖滃緤閻ｅ本宕查悗锝庡枟閻撳倹绻濇繝鍌滃闁绘帞绮幈銊ノ熺紒妯荤€繝銏ｎ潐濞茬喖寮诲☉妯滄梹鎷呴崷顓фФ闂佽瀛╅崙褰掑储閽樺鍤楅柛鏇ㄥ灠缁€瀣亜閹捐泛鏋庨柍?
        result = np.zeros_like(arr)
        for c in range(arr.shape[0]):
            result[c] = _apply_kspace_degradation_and_rician_noise(arr[c:c+1], sigma, k_factor)[0]
        return result
    
    # 闂傚倷绀侀幉锟犮€冮崱妞曟椽鎮㈤悡搴ｅ姦濡炪倖甯婄粈渚€宕崫鍔界懓顭ㄩ崼銏㈡毇闂佽桨绀佺粔鎾綖閵忕姭鏋栧☉?shape [1, H, W]
    slice_2d = arr[0]  # [H, W]
    
    # --- 1. K缂傚倸鍊风粈渚€寮甸鈧—鍐寠婢光晜鐩畷绋课旈埀顒傜不閿濆棎浜滈煫鍥ㄦ尰閸ｆ椽鏌?(婵犵數濮烽。浠嬪焵椤掆偓閸熷潡鍩€椤掆偓缂嶅﹪骞冨Ο鑽ょ畽闁活噮鎽ctor < 1.0) ---
    if k_factor < 1.0:
        # FFT闂傚倷绀侀幉锛勬暜閻愬樊鐎堕柤娴嬫櫇绾惧ジ鎮橀悙闈涗壕闁靛洦绻冩穱?        kspace = np.fft.fftshift(np.fft.fft2(slice_2d))
        
        # 闂傚倷绀侀幉锛勬暜濡ゅ啰鐭欓柟瀵稿Х绾句粙鏌熼崜褏甯涢柛搴㈩殜閺屽秵娼悧鍫偘闂佺硶鏅涢惌鍌炲蓟閻旂儤鍋橀柍鈺佸暞閻濇牠姊?(婵犵數鍎戠徊钘壝洪敂鐐床闁告洦鍨板Ч鏌ユ煃瑜滈崜娆撳煡婢舵劕绠奸柛鎰╁妼缁犲湱绱撴担瑙勩仢缂佺姵鐗曢悾鐑芥偐鐠囪弓绱堕梺鍛婃处閸樺ジ鐛?
        mask = np.zeros(kspace.shape, dtype=np.float32)
        center_y, center_x = kspace.shape[0] // 2, kspace.shape[1] // 2
        half_width_y = int(kspace.shape[0] * k_factor / 2)
        half_width_x = int(kspace.shape[1] * k_factor / 2)
        
        mask[center_y - half_width_y:center_y + half_width_y,
             center_x - half_width_x:center_x + half_width_x] = 1.0
        
        # 闂備礁婀遍崢褔鎮洪妸銉綎濠电姵鑹鹃弸渚€鏌曢崼婵愭Ц缂佺姵濞婇弻宥夊传閸曨偅娈堕梺?
        kspace_truncated = kspace * mask
        
        # Inverse FFT back to image space after truncation.
        slice_low_res = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_truncated)))
        slice_low_res = slice_2d
    
    # --- 2. 濠电姷鏁搁崕鎴犵礊閳ь剚銇勯弴鍡楀閸欏繑绻濋悽娈垮毀cian闂傚倷绶氬鑽ょ礊閸℃顩叉繝濠傜墛閺?---
    # 闂傚倷鐒﹂惇褰掑垂婵犳艾绐楅柟鐗堟緲閸ㄥ倹鎱ㄥ鍡楀箻闁崇粯妫冮弻鏇㈠醇濠靛棭浼冮梺鎼炲€曢澶愬蓟閿濆應鏋庢俊顖氭惈椤偊姊虹紒妯肩畺闁烩晩鍨堕獮鍡涘礃椤旇偐顦板銈嗙墬缁矂鎷忕€ｎ喗鈷戦柛婵嗗娴滅偤鏌涢妸锔界凡闁宠绉瑰顕€鍩€椤掑倻鍗氶柣鏃傚帶缁狙囨煙缁嬫寧鎹ｉ柣?
    noise1 = np.random.normal(0, sigma, slice_low_res.shape)
    noise2 = np.random.normal(0, sigma, slice_low_res.shape)
    
    # Rician闂傚倷绶氬鑽ょ礊閸℃顩叉繝濠傜墛閺? sqrt((signal + noise1)^2 + noise2^2)
    noisy_real = slice_low_res + noise1
    noisy_imag = noise2
    slice_rician = np.sqrt(noisy_real**2 + noisy_imag**2)
    
    # --- 3. 闂備浇宕甸崰宥咁渻閹烘梻鐭嗗ù锝呮贡閻濆爼鏌嶈閸撶喖寮诲☉銏犵闁哄啠鍋撻柣?, 1]闂傚倷娴囬崑鎰板储瑜斿畷顖烆敃閳?---
    slice_final = np.clip(slice_rician, 0.0, 1.0)
    
    # 婵犵數濮烽。浠嬪焵椤掆偓閸熷潡鍩€椤掆偓缂嶅﹪骞冨Ο璇茬窞闁归偊鍓涢悡鎴濐渻閵堝懐绠伴悗姘煎弮閻擃剟顢楅崟顒傚幈闂佽鍎崇壕顓熺墡婵犵數鍋涢幊蹇涙偡閳轰胶鏆︽繝闈涱儐閸嬫劙鏌涘▎蹇ｆШ鐟滃府缍佸娲箰鎼达絺妲堥梺鍏兼た閸ㄨ泛鐣疯ぐ鎺濇晬闁绘劖娼欐禒娲⒑闂堟单鍫ュ疾濠婂牞缍栫€光偓閸曨剛鍘辨繝鐢靛Т閹冲孩绂嶈ぐ鎺撶厸濞达綀顫夊畷灞绢殽閻愬樊鍎旀鐐叉喘椤㈡﹢鎮欓崜渚囨Ч濠电姷顣藉Σ鍛村垂椤忓嫀锝夊礋椤撶噥娼熷┑鐘绘涧濞层劑銆呴悜鑺ョ厱婵炴垵宕獮妤呮椤掑澧撮柡宀€鍠撻幃浼村焺閸愨晝褰嗛梺鐓庣亪閺呯娀寮诲☉銏犲唨鐟滃酣骞嗛崼銉︾厽闁规儳顕幊鍕煙瀹勭増鍣介柟鐟板婵℃悂鏁傞懞銉悅 slice_final闂傚倷鐒︾€笛呯矙閹达附鍎楀〒姘ｅ亾鐎规洘娲栬灃闁告侗鍘鹃弻褍顪冮妶鍡欏闁荤啙鍛笉濞寸姴顑嗛悡鐔兼煥濠靛棙顥犻柟鍐插暣閹粙顢涢敐鍛凹闂?slice_2d
    if foreground_only:
        if fg_threshold is None:
            tmin = float(np.min(slice_2d))
            tmax = float(np.max(slice_2d))
            threshold = (tmin + tmax) / 2.0
        else:
            threshold = float(fg_threshold)
        mask = (slice_2d > threshold).astype(np.float32)
        slice_final = mask * slice_final + (1.0 - mask) * slice_2d

    return slice_final[np.newaxis, ...]  # 闂傚倷鐒﹂幃鍫曞磿閹惰棄纾绘繛鎴欏灩閺?[1, H, W]

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None, 
                 foreground_crop=True, fg_threshold=None, fg_ratio_threshold=0.15,
                 max_attempts=20):
        """
        Args:
            dataset: 闂傚倷娴囧銊╂嚄閼稿灚娅犳俊銈傚亾闁伙絽鐏氱粭鐔煎焵椤掆偓椤?            inp_size: patch婵犵數濮伴崹褰掓偉閵忋倕绀冩い蹇撴噽濡?
            augment: 闂傚倷绀侀幖顐も偓姘卞厴瀹曡瀵奸弶鎴犵暰婵炴挻鍩冮崑鎾垛偓娈垮櫘閸嬪﹪骞冭瀹曠厧鈹戦崼娑樹喊婵犵數濮幏鍐川椤撴繄鎹曢柣?            sample_q: 闂傚倸鍊烽悞锕併亹閸愵煁娲晝閸屾稑浜遍梻渚囧墮缁夌敻宕愰悜鑺ュ仭婵炲棗绻愰顏呯箾閸繄鍩ｆ慨?            foreground_crop: 闂傚倷绀侀幖顐も偓姘卞厴瀹曡瀵奸弶鎴犵暰婵炴挻鍩冮崑鎾垛偓瑙勬穿缂嶄線銆侀弮鍫濈妞ゆ帒鍟幆鍫ユ⒒娴ｅ憡鍟為柟绋挎憸缁棃妫冨☉鎺撶€哄┑鐐叉閹稿摜绮堝畝鍕厱婵犻潧妫楅顐︽煕濡吋鏆柟顕€鏀遍幏鍛槹鎼搭喗袦濠电姰鍨归崢婊堝磻?
            fg_threshold: 闂傚倷绀侀幉锟犲箰閸濄儳鐭撻梻鍫熺▓閺嬪秵绻涢崱妯诲鞍闁抽攱鎹囬弻娑㈩敃閿濆洤顩梺绋款儐閹告儳鈽夐崹顐Ч閹艰揪缍嗗Σ鐩ne闂傚倷绀侀幉锛勬暜濡ゅ懏鍋￠柕濞炬櫅閸氳绻濇繝鍌滃缂佲偓婢舵劖鐓忓┑鐘叉捣閸樻盯鏌涘锝呬壕缂傚倸鍊烽懗鑸垫叏閻㈢桅闁绘劕鐏氱€?min+max)/2闂?            fg_ratio_threshold: patch婵犵數鍋為崹鍫曞箹閳哄懎鍌ㄩ柛蹇撳悑濞呯娀鏌ｅΟ娆惧殭闁汇倗鍋撻妵鍕冀閵娿劌顥濋梺鍛婃崄閸嬫劗妲愰幒鎾寸秶闁宠桨绲块姀鈶╁亾鐟欏嫭绀冩繝銏☆焾椤ｉ箖姊洪崫鍕殭闁稿﹤缍婇幃楣冩焼瀹ュ棛鍘搁梺鍓插亝閼圭偓绂嶆ィ鍐╃厽闊洦鎸剧粻鎵磼缂佹ê濮嶉柛鈹惧亾濡炪倖甯掗崐鍫曞焵椤掆偓椤戝洤危?-1闂傚倷鐒︾€笛呯矙閹次层劑鍩€椤掑倻纾奸弶鍫涘妿缁犳牠鏌熷畡鐗堝櫤闁诡垱鏌ㄩ埞鎴﹀醇閿濆拋妫熷┑鐘愁問閸犳缂撻崸妤€纾归柡鍥ュ灩閻掑灚銇勯幒鎴濐仾闁挎稓鍠撶槐鎺楀矗濡搫绁梺鍝勮嫰閼活垶顢樻總绋垮耿婵☆垳鍘х敮搴ㄦ⒒閸屾瑧鍔嶇憸鏉垮暙鐓ら柨鏇炲€归崑?            max_attempts: 闂傚倷绀侀幖顐︽偋閸愵喖纾婚柟鐐墯閻斿棝鏌涢銏☆棞闁哥喐婢樿闁挎洖鍊归崑锝夋煕閵夛絽濡界紒鈧崘鈹夸簻闁归偊鍏欓崑銏⑩偓娈垮櫘閸嬪棝寮鈧、娆撳礈瑜滃Σ鐑芥⒑閻戔晛澧查悽顖涘浮楠炲﹪骞橀懜闈涘簥闂佸憡鍔﹂崰鏍矆閸℃绠鹃柛鈩兩戠亸鐗堛亜鎼淬垻澧辩紒杈ㄥ浮閹亜鈻庤箛鎾亾閻ｅ瞼鐭嗛悗锝庡枟閳锋帡鏌涘☉鍗炲箻缂佺姴寮剁换婵婎槷闁稿鎹囬弻锝夋偄閼告妯傜紓浣插亾濞达絽婀遍悵鍫曟煃?
        """
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.foreground_crop = foreground_crop
        self.fg_threshold = fg_threshold
        self.fg_ratio_threshold = fg_ratio_threshold
        self.max_attempts = max_attempts

    def _get_sample_name(self, idx):
        current = self.dataset
        while hasattr(current, 'dataset'):
            current = current.dataset
        if hasattr(current, 'lr_dataset') and hasattr(current.lr_dataset, 'filenames'):
            filenames = current.lr_dataset.filenames
            if filenames:
                return filenames[idx % len(filenames)]
        if hasattr(current, 'dataset_1') and hasattr(current.dataset_1, 'filenames'):
            filenames = current.dataset_1.filenames
            if filenames:
                return filenames[idx % len(filenames)]
        if hasattr(current, 'filenames'):
            filenames = current.filenames
            if filenames:
                return filenames[idx % len(filenames)]
        return f'index={idx}'

    def _validate_pair_shapes(self, idx, img_lr, img_hr, img_mask=None):
        h_lr, w_lr = img_lr.shape[-2:]
        h_hr, w_hr = img_hr.shape[-2:]
        sample_name = self._get_sample_name(idx)

        if min(h_lr, w_lr, h_hr, w_hr) <= 0:
            raise ValueError(
                f'Invalid non-positive shape for sample {sample_name}: '
                f'LR={tuple(img_lr.shape)}, HR={tuple(img_hr.shape)}'
            )

        if h_hr < h_lr or w_hr < w_lr:
            raise ValueError(
                f'HR is smaller than LR for sample {sample_name}: '
                f'LR={tuple(img_lr.shape)}, HR={tuple(img_hr.shape)}'
            )

        if h_hr % h_lr != 0 or w_hr % w_lr != 0:
            raise ValueError(
                f'LR/HR shapes are not integer multiples for sample {sample_name}: '
                f'LR={tuple(img_lr.shape)}, HR={tuple(img_hr.shape)}'
            )

        scale_h = h_hr // h_lr
        scale_w = w_hr // w_lr
        if scale_h != scale_w:
            raise ValueError(
                f'LR/HR height-width scales differ for sample {sample_name}: '
                f'LR={tuple(img_lr.shape)}, HR={tuple(img_hr.shape)}, '
                f'scale_h={scale_h}, scale_w={scale_w}'
            )

        if self.inp_size is not None and (h_lr < self.inp_size or w_lr < self.inp_size):
            raise ValueError(
                f'LR patch size {self.inp_size} is larger than sample {sample_name}: '
                f'LR={tuple(img_lr.shape)}, HR={tuple(img_hr.shape)}'
            )

        return scale_h

    def _align_mask_to_hr(self, idx, img_mask, img_hr):
        if img_mask is None or tuple(img_mask.shape[-2:]) == tuple(img_hr.shape[-2:]):
            return img_mask

        sample_name = self._get_sample_name(idx)
        orig_dtype = img_mask.dtype
        aligned_mask = F.interpolate(
            img_mask.unsqueeze(0).to(torch.float32),
            size=img_hr.shape[-2:],
            mode='nearest',
        ).squeeze(0).to(orig_dtype)
        print(
            f'Warning: resized mask to match HR for sample {sample_name}: '
            f'mask {tuple(img_mask.shape)} -> {tuple(aligned_mask.shape)}, '
            f'HR={tuple(img_hr.shape)}'
        )
        return aligned_mask

    def __len__(self):
        return len(self.dataset)
    
    def _get_foreground_crop_position(self, img_lr, h_lr, w_lr):
        """
        闂傚倷绶氬鑽ゆ嫻閻旂厧绀夐悘鐐靛亾濞呯娀鏌ｅΟ娆惧殭闁汇倗鍋撻妵鍕冀閵娿劌顥濋梻鍌氭噺閿曘垽寮诲☉銏犵厸闁告劑鍔嶉悘浣糕攽閻愯尙澧涢柛銊ㄦ硾椤洩绠涘☉妯诲祶濡炪倖鎸鹃崰鎰版偟椤忓嫧鏀介柍鈺佸枤濞堟梹淇婇锝庢疁鐎规洦鍓熼獮姗€顢欑憴锝嗗瘲闂傚鍋勫ú锕€顫忛崷顓″С婵鍩栭崑鈥趁归敐澶樻妞わ讣绠戦…鍧楀礂婢跺﹣澹曢梻鍌欑劍鐎笛呯矙閹寸偟闄勯柡鍐ㄥ€瑰畷鍙夋叏濡寧纭剧紒鐘虫緲闇夐柨婵嗘噹椤ュ繑淇婇姘煎剶闁哄备鈧剚鍚嬪璺侯儐閺嗙娀姊?        
        Returns:
            (x0, y0): 闂佽楠哥紞濠傤焽閼姐倗纾芥慨妯夸含缁犳棃鏌涚仦鍓х煁闁哥姴妫濋弻鐔兼倻濡櫣浠哥紓鍌氱Т缁绘﹢寮婚垾鎰佸悑闁告侗鍠氶妶顐ょ磽娴ｄ粙鍝虹紒璇插暞缁岃鲸绻濋崶褏锛滃┑鐐村灦閻噣宕欐禒瀣拺闁告繂瀚晶鍗灻瑰鍕畺缂佹梻鍠庨～婊堝焵椤掆偓閻ｅ嘲顫濈捄铏归獓闂佸湱顭堟鎼佸焵椤戞儳鈧繂顫忓ú顏嶆晝闁靛繒濮崑鎾搭槹鎼达絿顦梺鍛婁緱閸ㄥ磭澹曢悙顑句簻闁哄啫娲︾涵鍓佺磼閻樺磭鍙€闁哄本鐩俊鐑芥晜閽樺绶畂ne
        """
        # Compute the foreground mask on the LR image.
        img_lr_np = img_lr.cpu().numpy() if isinstance(img_lr, torch.Tensor) else img_lr
        
        # 婵犵數濮烽。浠嬪焵椤掆偓閸熷潡鍩€椤掆偓缂嶅﹪骞冨Ο璇茬窞闁归偊鍓欏宄邦渻閵堝棛澧柛鎴犳嚀铻ｉ悗锝庡枟閳锋垶銇勯幇鍓佸埌闁告繂鎼湁婵犲﹤鐗忛悾娲煛娴ｅ摜肖濞寸媴绠撻幐濠冨緞鐏炴儳浠圭紓鍌氬€烽悞锕傘€冭箛娑樼婵炴垶姘ㄩ崡姘舵倵濞戞瑯鐒介柍缁樻⒐閵囧嫯绠涢幘璺侯暤闂佺顑嗛幐鎼侊綖濠靛瀚夐梻鍫熶緱濡粓姊绘担鍦菇闁告柨鐭傚畷娲箵绾潣
        if mask.ndim == 3:
            mask_2d = mask[0]
        else:
            mask_2d = mask
            
        # 闂傚倷鑳堕幊鎾绘倶濮樿泛纾块柟鎯版閺勩儳鈧厜鍋撻柛鏇ㄥ亞椤撳ジ姊洪柅鐐茶嫰婢у鈧鍠栭悘姘嚗閸曨剛绡€闁稿被鍊栭悗顐︽⒒娴ｅ憡鎯堥柟宄邦儔瀹曡瀵奸弶鎴狅紵濠殿喗顭堟ご鎼侊綖閺囥垺鐓熸俊顖濇娴犳盯鏌ｉ敐澶夋喚闁哄本绋栭ˇ鎶芥煕閺冣偓閻楃娀骞?        fg_coords = np.argwhere(mask_2d > 0.5)  # [N, 2] 闂傚倷绀侀幖顐ょ矓閸洖鍌ㄧ憸蹇撐? (x, y)
        
        if len(fg_coords) == 0:
            # 濠电姷鏁搁崑娑欏緞閸ヮ剙绀堟繝闈涙４閼板灝銆掑锝呬壕閻庤娲樼划宀勵敇婵傜骞㈡俊顖滃帶鐠佹煡姊绘担鐟邦嚋缂佸鍨块幆灞炬媴閻戞ê搴婇梺纭呮彧缁犳垿鎮欐繝鍐瘈闁汇垺顔栧顤磂婵犵數鍋犻幓顏嗙礊閳ь剚绻涙径瀣鐎殿噮鍋婃俊鑸靛緞鐎Ｑ勫瘲闂傚鍋勫ú锕€顫忛崷顓″С婵鍩栭崑鈥趁归敐澶樻妞わ讣绠戦…鍧楀礂婢跺﹣澹?
            return None
        
        # 闂傚倷绀侀崥瀣磿閹惰棄搴婇柤鑹扮堪娴滃綊鏌涢妷顔煎缂佲偓閸懇鍋撻獮鍨姎婵☆偅鐟╅幃鐢割敍閻愬鍘遍梺鍝勮癁閸曨剙鍓甸梻浣虹帛閹稿鎮烽埡鍛畾闁告劦鍠栫粈瀣亜閺傚灝鈷旂憸浼寸畺濮婄儤瀵煎▎鎴濆煂闂佸搫鐗滈崜鐔煎Υ?        x_min = fg_coords[:, 0].min()
        x_max = fg_coords[:, 0].max()
        y_min = fg_coords[:, 1].min()
        y_max = fg_coords[:, 1].max()
        
        # 闂備浇宕垫慨宕囨閵堝洦顫曢柡鍥ュ灪閸嬧晛鈹戦悩瀹犲缂佲偓閸懇鍋撻獮鍨姎婵☆偅鐟╅幃鐢割敍閻愬鍘遍梺鍝勮癁閸曨剙鍓甸梻浣虹帛閹稿鎮烽埡鍛畾闁告劦鍠栫粈瀣亜閹板墎绋绘い顐㈡嚇閺?        fg_height = x_max - x_min + 1
        fg_width = y_max - y_min + 1
        
        # 婵犵數濮烽。浠嬪焵椤掆偓閸熷潡鍩€椤掆偓缂嶅﹪骞冨Ο璇茬窞闁归偊鍓涢ˇ顓㈡倵楠炲灝鍔氭俊顐ｇ懇閹敻顢涢悙瀵稿幈闂佸搫璇為崟顒€鍓甸梻浣虹帛閹稿鎮疯缁傚秴顫㈠畝鈧悿鈧柟鑹版彧缂嶁偓濠㈣锕㈤弻锝夋偄閼告妯傜紓浣插亾濞达絽婀遍悵鍫曟煃瑜滈崜鐔煎箖濡法鐤€闁规儳鐡ㄩ崕鎾愁渻閵堝棙澶勯柛娆忓暣瀵偄顓奸崨顖涙畷闂佸憡鍔︽禍鐐侯敊婢舵劖鈷戦柟鑲╁仜閸旀粓鏌熸潏鈺婄吇ne婵犵數鍋犻幓顏嗙礊閳ь剚绻涙径瀣鐎殿噮鍋婃俊鑸靛緞鐎Ｑ勫瘲闂傚鍋勫ú锕€顫忛崷顓″С婵鍩栭崑鈥趁归敐澶樻妞わ讣绠戦…鍧楀礂婢跺﹣澹?
        if fg_height < h_lr or fg_width < w_lr:
            return None
        
        # 闂傚倷绶氬鑽ゆ嫻閻旂厧绀夐悘鐐靛亾濞呯娀鏌ｅΟ娆惧殭闁汇倗鍋撻妵鍕冀閵娧呯暭闂佺粯绻傞悥濂稿蓟閿濆鏅查柛娑卞枟閹峰崬顪冮妶鍐ㄧ仼妞ゆ垵顦悾宄邦潨閳ь剙鐣烽妸褉鍋撳☉娆欎緵闁告艾鍊垮娲传閸曨厼鈪电紓鍌氬€瑰畝姝屾＂闂佺硶鍓濈粙鎴犵矆閸喆浜滈煫鍥ㄦ尰閹癸絿绱掗埀顒佸緞閹邦厼鈧爼鐓崶銊︹拻缂佺姵宀搁弻锝夋晜鐠囪尙浠悗鍨緲鐎氼剟锝炲┑瀣垫晣鐟滃秵绂掔憴鍕閻庣數顭堣ⅷ闂佺瀛╂竟鍡涘箲閵忕媭娼╅弶鍫涘妼閻濈増绻涙潏鍓ф偧妞ゎ厼鐗撹矾闁告稑鐡ㄩ悡娑㈡煕濠娾偓缁€渚€鎮橀敓鐘崇厽闁挎棁顕ч弸鐔虹磼?        best_x0, best_y0 = None, None
        best_fg_ratio = 0
        
        for attempt in range(self.max_attempts):
            # 闂傚倷绶氬鑽ゆ嫻閻旂厧绀夐悘鐐靛亾濞呯娀鏌ｅΟ娆惧殭闁汇倗鍋撻妵鍕冀閵娧呯暭闂佺粯绻傞悥濂稿蓟閿濆鏅查柛娑卞枟閹峰崬顪冮妶鍐ㄧ仼妞ゆ垵顦悾宄邦潨閳ь剙鐣烽妸褉鍋撳☉娆欎緵闁告艾鍊垮娲传閸曨厼顤€濠碉紕鍋樼划娆撳春閳ь剚銇勯幒鍡椾壕濡炪倧闄勬竟鍡涘焵椤掍礁鎼搁柛鏂块叄楠炴劘顦虫い锔惧閹棃鍨鹃幓鎺戭嚙
            x0 = random.randint(int(x_min), int(max(x_min, x_max - h_lr + 1)))
            y0 = random.randint(int(y_min), int(max(y_min, y_max - w_lr + 1)))
            
            # Measure how much foreground the candidate patch contains.
            patch_mask = mask_2d[x0:x0+h_lr, y0:y0+w_lr]
            
            # Track the best foreground crop seen so far.
            if fg_ratio > best_fg_ratio:
                best_fg_ratio = fg_ratio
                best_x0, best_y0 = x0, y0
            # Return early once the foreground ratio passes the threshold.
            if fg_ratio >= self.fg_ratio_threshold:
                return (x0, y0)
        # Fall back to the best crop if no attempt met the threshold.
        if best_x0 is not None and best_fg_ratio > 0:
            return (best_x0, best_y0)
        # 闂備浇顕уù鐑藉箠閹剧粯鍤愭い鏍仜閻鐓崶銊﹀皑闁哄矉绠撻弻宥夊煛娴ｅ憡娈茬紓浣哄У鐢繝寮诲☉銏犵鐎规洖娉﹂敐鍡欑闁告侗鍠氶幊鍥煛娴ｅ摜孝闁伙絾绻堝畷姗€顢旈崱鈺佹暭闂傚倷鐒﹂幃鍫曞磿閺屻儱绠炵紒灞剧崲e婵犵數鍋犻幓顏嗙礊閳ь剚绻涙径瀣鐎殿噮鍋婃俊鑸靛緞鐎Ｑ勫瘲闂傚鍋勫ú锕€顫忛崷顓″С婵鍩栭崑鈥趁归敐澶樻妞わ讣绠戦…鍧楀礂婢跺﹣澹?
        return None

    def __getitem__(self, idx):
        raw = self.dataset[idx]
        if isinstance(raw, (tuple, list)) and len(raw) == 3:
            img_lr, img_hr, img_mask = raw
            has_mask = True
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            img_lr, img_hr = raw
            img_mask = None
            has_mask = False
        else:
            raise ValueError(
                f'sr-implicit-paired expects dataset item as (lr, hr) or (lr, hr, mask), got type={type(raw)}'
            )

        s = self._validate_pair_shapes(idx, img_lr, img_hr, img_mask if has_mask else None)
        if has_mask:
            img_mask = self._align_mask_to_hr(idx, img_mask, img_hr)
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
            if has_mask:
                crop_mask = img_mask[:, :h_lr * s, :w_lr * s]
        else:
            w_lr = self.inp_size
            h_lr = self.inp_size
            
            # Prefer crops that overlap the foreground when requested.
            if self.foreground_crop:
                if crop_pos is not None:
                    x0, y0 = crop_pos
                else:
                    # Fallback to random cropping if no foreground crop is found.
                    x0 = random.randint(0, img_lr.shape[-2] - h_lr)
            else:
                # 闂傚倷绀侀幉锟犫€﹂崶顒€绐楅幖鎼厜缂嶆牠鏌熼幍顔碱暭闁抽攱娲熼悡顐﹀炊閵婏腹鎷圭紓浣靛妽瀹€鎼佸箖闁垮濯村瀣瘨濡差喗绻濋姀鐘插辅闁?
                x0 = random.randint(0, img_lr.shape[-2] - h_lr)
                y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            
            crop_lr = img_lr[:, x0: x0 + h_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            h_hr = h_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + h_hr, y1: y1 + w_hr]
            if has_mask:
                crop_mask = img_mask[:, x1: x1 + h_hr, y1: y1 + w_hr]
            
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            if has_mask:
                crop_mask = augment(crop_mask)

        if crop_lr.shape[-2] == 0 or crop_lr.shape[-1] == 0 or crop_hr.shape[-2] == 0 or crop_hr.shape[-1] == 0:
            raise ValueError(
                f'Empty crop generated for sample {self._get_sample_name(idx)}: '
                f'crop_lr={tuple(crop_lr.shape)}, crop_hr={tuple(crop_hr.shape)}'
            )
        if has_mask and (crop_mask.shape[-2] == 0 or crop_mask.shape[-1] == 0):
            raise ValueError(
                f'Empty mask crop generated for sample {self._get_sample_name(idx)}: '
                f'crop_mask={tuple(crop_mask.shape)}'
            )

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        result = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': crop_hr,
            'scale': s,
        }
        if has_mask:
            result['mask'] = crop_mask
        return result


def resize_fn(img, size):
    """
    闂備浇宕垫慨鎾敄閸涙潙鐤い鏍仜濮规煡鏌ㄥ┑鍡╂Ц闁绘劕锕弻鐔碱敍閸℃鏆欏瑙勭墬缁绘繈濮€閻橀潧鐏梺鍛婂姦娴滄繈鎯侀幘缁樷拺闁圭娴烽埥澶愭煛閸偄澧寸€殿噮鍋婇、姘跺焵椤掑嫮宓侀柟閭﹀幗婵绱掔€ｎ厽纭舵い锔垮嵆濮婄粯绗熼崶褌绨奸悗鍏夊亾闁告稑锕﹂々鎻捨旈敐鍛殲闁哄拋鍓熼弻娑㈡晜缂佹ɑ鍟營闂傚倷鐒︾€笛呯矙閹次诲洭顢涢悙鑼唵闂佺粯顨呴悧鍡欐閻愬瓨鍙忔俊顖濇娴滎亪鏌涚€ｎ偅灏い顐ｇ箞瀵敻妫冨☉杈棥闂傚倷鐒︾€笛呯矙閹存繐鑰块柕鍠瑩姊绘担鐟邦嚋缂佸鍨胯棟妞ゆ牗顕㈤悷閭︽建闁逞屽墮閻?    Args:
        img: Tensor [C, H, W] 闂?PIL Image
        size: tuple (H, W) 闂?int
    Returns:
        Tensor [C, H, W]
    """
    # 闂傚倷鑳堕崕鐢稿疾閳哄懎绐楁俊銈呮噺閸嬪鏌ㄥ┑鍡╂Ч闁哄拋鍓氶幈銊ヮ潨閸℃ぞ绨婚梺浼欓檮缁捇骞冮悜钘夊嵆婵ê宕俊钘夆攽閻橆偄浜鹃梺鎼炲労閸撴瑧绮堟径鎰厓闁靛鍎遍弳杈╃磼閳ь剛鈧綆鍠楅埛鎴炪亜閹板墎鍒伴柛婵嗘惈闇?MRI 闂?npy 闂傚倷娴囧銊╂嚄閼稿灚娅犳俊銈傚亾闁伙絽鐏氱粭鐔煎焵椤掑嫬鏋佺€广儱顦柨銈夋煙妫版繃鐓廰t闂傚倷鐒︾€笛呯矙閹次层劑鍩€椤掑倻纾奸弶鍫涘妿缁犳ê霉濠婂啯鍟炵紒鍌涘笧閳ь剨缍嗘禍婵嬵敂閿熺姵鈷掑〒姘搐婢ь噣鏌ｈ箛鏃傜疄鐎规洏鍨芥俊鑸靛緞婵犲嫷鍞村┑鐘垫暩婵挳宕姘ｆ灁闁割偅娲橀悡娆撴偣閸ワ絺鍋撻搹顐や粚婵犵數鍋炲娆徫涘┑瀣祦闁硅揪璁ｇ紞鏍ㄣ亜閹垮嫮纾挎い?PIL 闂備浇顕х花鑲╁緤缂佹鐝堕柛顐犲劚閸氬湱鎲搁弮鍫濊摕鐎光偓閸愵亞鏉搁梺鍦亾閸撴岸宕?缂傚倸鍊搁崐鎼佸磹閻㈢鐤炬繝濠傜墕濡?    # 闂傚倷娴囬妴鈧柛瀣尰閵囧嫰寮介妸褉妲堥梺浼欏瘜閸ｏ絽顕ｉ崼鏇為唶闁绘梻顭堢喊宥夋⒑閸涘﹦鎳冩俊顐ｇ洴閹噣宕滆鐎氭岸鏌ょ喊鍗炲幋闁稿鎹囬幃婊堟嚍閵夛附鐝繝娈垮枟缁诲嫬顪冮崓鐬.ndarray, torch.Tensor, PIL.Image

    # 闂備浇宕甸崰鎰版偡閵壯€鍋撳鐓庣仯闁?size 婵?(H, W)
    if isinstance(size, int):
        size = (size, size)

    # Convert PIL images to tensors before resizing to keep a unified code path.
    if isinstance(img, Image.Image):
        tensor = transforms.ToTensor()(img)
        # numpy array: 闂傚倷绀侀幉锟犳偡椤栫偛鍨傞柣銏㈩焾缁€鍌涙叏濡炶浜鹃悗?(H,W) 闂?(C,H,W)
        arr = img
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[2] in (1, 3) and arr.shape[0] not in (1, 3):
            # 婵犵數濮伴崹鐓庘枖濞戞埃鍋撳鐓庢珝妤?HWC -> CHW
            arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr.copy())
        # 缂傚倷鑳堕搹搴ㄥ矗鎼淬劌绐楅柡鍥╁У瀹曞弶鎱ㄥ鍡楀箻闁?float
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
    elif isinstance(img, torch.Tensor):
        tensor = img.clone()
    else:
        tensor = transforms.ToTensor()(img)

    # Ensure CHW layout.
    # Ensure CHW layout.
        tensor = tensor.unsqueeze(0)

    # 闂佽瀛╅鏍窗閺嶎厼绠规い鎰剁畱閺勩儲淇婇妶鍕厡闁?float32闂傚倷鐒︾€笛呯矙閹寸偟闄勯柡鍐ㄥ€瑰畷鍙夋叏濡灝鐓愰柛瀣ф櫆缁绘盯宕卞Ο鍝勵潔濠电偛鐗嗛悥濂稿蓟濞戞瑧绡€闁告侗鍙庡Λ鍫ユ煟鎼达紕浠涢柣妤冨Т閻ｇ兘鎳滈崹顐ょФ闂佸憡鎸嗛崪浣剐熸繝鐢靛仦閸ㄥ爼骞愰崫銉х煋闁绘垵顫曢埀顒€鎳橀弫鍌炴煥椤栨矮澹曟繛杈剧到閸犳艾顭囬幇鐗堢厸闁糕剝鐟ラ弸娑欘殽閻愭潙鐏存い銏＄懇濮婅崵鈧灚鎮傞埞蹇撯攽閳藉棗浜炲褎顨堥幑銏ゅ磼閻愬牆娲鎷岀疀鐎ｂ晜鐫忛梻浣烘嚀閹碱偆绮旈崼鏇ㄦ晜闂侇剙绉甸悡鏇㈡煛瀹ュ啫濡垮鐟板缁?    tensor = tensor.to(dtype=torch.float32)

    # 闂傚倷绀佸﹢杈╁垝椤栨粍鏆滈柟鐑橆殔閻掑灚銇勯幒鍡椾壕闂侀潧鐗忛…鍫ュΥ閹烘绀堝ù锝囨嚀閺?4D: [N, C, H, W]
    need_squeeze = False
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        need_squeeze = True

    # 婵犵數鍋犻幓顏嗙礊閳ь剚绻涙径瀣鐎?PyTorch 闂?interpolate闂傚倷鐒︾€笛呯矙閹达附鍋嬮柛娑卞灡瀹曞弶鎱ㄥΟ鍨厫闁?float 缂傚倸鍊风欢锟犲窗濡ゅ懎纾跨€规洖娲犻崑鎾舵喆閸曨偀鏋欓梺杞扮缁夐潧顕ラ崟顐熸婵﹢纭稿Σ绌奿cubic 闂傚倷绀佸﹢杈╁垝椤栨粍鏆滈柟鐑橆殔閻?    # align_corners=False 闂備浇顕уù鐑藉极婵犳艾鐒垫い鎺嶈兌閵嗘帡鏌よぐ鎺旂暫婵﹤顭峰畷濂稿閻樿鲸鍕冮梻浣告惈濡绱炴担鍓插殨妞ゆ劧绠戠痪褔骞栫€涙ɑ灏ù婊冨⒔閳ь剙绠嶉崕鍗灻洪妸褎顫曟慨妯垮煐閻撴盯鏌嶈閸撴岸濡堕敐澶婄闁挎繂鎳嶆竟鏇炩攽椤旀枻渚涢柛鎾寸洴瀹曞搫螖閸涱喚鍘搁梺鎼炲劀鐏炶姤顔嶉梺鍏煎閻熲晠骞?    resized = F.interpolate(tensor, size=(int(size[0]), int(size[1])), mode='bicubic', align_corners=False)

    if need_squeeze:
        resized = resized.squeeze(0)

    return resized

@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, batch_per_gpu=16,
                 add_noise=False, noise_sigma=0.0, noise_k_factor=1.0, noise_mode='rician',
                 foreground_only=False, fg_threshold=None):
        """
        Wrapper that downsamples HR to LR. Optional: add noise to the LR patch.

        Args:
            dataset: underlying dataset returning HR images (torch Tensor CHW or numpy)
            inp_size: target LR patch size (int) or None
            scale_min/scale_max: scale sampling range
            augment: boolean flip/transpose augmentation
            sample_q: unused here
            batch_per_gpu: reuse scale per this many calls
            add_noise: whether to add noise to LR patches
            noise_sigma: sigma for the added complex Gaussian noise
            noise_k_factor: K-space retention factor (0-1), <1 for low-res simulation
            noise_mode: 'rician' (default) or 'gaussian' (keeps real noise)
        """
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.last_s = random.uniform(self.scale_min, self.scale_max)
        self.batch_per_gpu = batch_per_gpu
        self.call_count = -2

        # Noise options for LR patches
        self.add_noise = add_noise
        self.noise_sigma = float(noise_sigma)
        self.noise_k_factor = float(noise_k_factor)
        self.noise_mode = noise_mode
        self.foreground_only = bool(foreground_only)
        self.fg_threshold = fg_threshold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        self.call_count += 1
        if self.call_count % self.batch_per_gpu == 0:
            s = random.uniform(self.scale_min, self.scale_max)
            self.last_s = s
        else:
            s = self.last_s

        # 闂傚倷娴囬妴鈧柛瀣尰閵囧嫰寮介妸褉妲堥梺浼欏瘜閸犳牠鍩ユ径鎰闁告鍋愰崑鎾澄旈崪鍐◤闂佸湱鍎ら〃鍛村几娓氣偓閺岋絽螖閳ь剟鎮ч崱娆戠焾闁挎洖鍊归埛鎺楁煕閳╁喚娈曟い銉磿缁辨帗锛愬┑鍡楃睄閻庤娲橀悷鈺呭箖濠婂吘鐔兼偂鎼搭喗顥欓梺璇插椤旀牠宕伴弽顭戞闊洦娲橀～?        #   img            闂傚倷鑳堕崑銊╁磿閺屻儱钃熼柨鐔哄Т閻?闂傚倷绀侀幖顐﹀箯鐎ｎ喖闂柨婵嗩槸閻?NpyFolder闂傚倷鐒︾€笛呯矙閹达附鍎楀ù锝囧劋瀹曟煡鏌熸潏鍓х暠闁绘劕锕弻锝夊箛椤撶偟绁风紓浣插亾閻庯綆鍏橀崑?HR 闂傚倷鐒﹂幃鍫曞磿閺屻儱绠繝闈涚墛椤洘绻濋棃娑卞剱妞ゃ儱妫濋弻宥堫檨闁告挾鍠栭獮濠傗槈閵忊晜鏅滈梺鎼炲劥閸╂牠寮埀顒€鈹戦悙鏉戠仸闁瑰憡鎮傞、鏍ㄥ緞閹邦剟妫锋繛鏉戝悑濞兼瑩宕橀埀?        #   (lr, hr, mask) 闂傚倷鑳堕崑銊╁磿閺屻儱钃熼柨鐔哄Т閻?PairedNpyFoldersWithMask闂傚倷鐒︾€笛呯矙閹达附鍎旈柣鎾崇瘍濞差亜閿ゆ俊銈傚亾缂佺姵濞婇弻鏇熷緞閸績鍋撻弴銏犵厺闊洦绋掗悡鐔搞亜椤愵偄澧┑顔煎€块弻娑樜熺粙搴撳亾濡ゅ懎鐒垫い鎺戝濞懷勭節閳ь剟宕￠悙鈺傜亖?LR/HR/mask
        raw = self.dataset[idx]
        if isinstance(raw, (tuple, list)) and len(raw) == 3:
            img_lr_raw, img, crop_mask = raw   # [1,H_lr,W_lr], [1,H_hr,W_hr], [1,H_hr,W_hr]
            has_mask = True
            has_paired_lr = True
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            img_lr_raw, img = raw
            crop_mask = None
            has_mask = False
            has_paired_lr = True
        else:
            img = raw
            crop_mask = None
            has_mask = False
            has_paired_lr = False

        if has_paired_lr:
            # ---- 闂傚倸鍊烽悞锕€顭垮Ο鑲╃煋闁圭虎鍣弫瀣亜閺囨浜鹃悗娈垮櫘閸嬪﹪骞冭瀹曠厧鈹戦崼娑樹喊闂傚倸鍊搁崐绋课涘Δ鈧灋婵犲﹤鐗婇崕妤呮煙椤栧棗鑻▓顐︽⒑閸涘﹥绀€濠⒀冩捣缁絽螖閸涱喚鍘介梺瀹犳〃閼冲爼骞楅悩缁樺仺妞ゆ牗鍑归崵鐔兼煙閸欏娈曢柟宄版噽缁瑥鈻庨悙顑芥寗闂備浇顕ч柊锝咁焽瑜忕槐鐐寸節閸曨厾顦梺鍓插亖閸庢煡宕?LR / HR / mask patch ----
            if self.inp_size is None:
                h_lr_full = img_lr_raw.shape[-2]
                w_lr_full = img_lr_raw.shape[-1]
                h_hr = round(h_lr_full * s)
                w_hr = round(w_lr_full * s)
                crop_hr = img[:,          :h_hr,     :w_hr]
                crop_lr = img_lr_raw[:,   :h_lr_full, :w_lr_full]
                if has_mask:
                    crop_mask = crop_mask[:, :h_hr, :w_hr]
            else:
                h_lr = math.floor(self.inp_size / s + 1e-9)
                w_lr = math.floor(self.inp_size / s + 1e-9)
                h_hr = round(h_lr * s)
                w_hr = round(w_lr * s)
                x0_lr = random.randint(0, img_lr_raw.shape[-2] - h_lr)
                y0_lr = random.randint(0, img_lr_raw.shape[-1] - w_lr)
                x0_hr = x0_lr * s
                y0_hr = y0_lr * s
                crop_hr = img[:, x0_hr: x0_hr + h_hr, y0_hr: y0_hr + w_hr]
                crop_lr = img_lr_raw[:, x0_lr: x0_lr + h_lr, y0_lr: y0_lr + w_lr]
                if has_mask:
                    crop_mask = crop_mask[:, x0_hr: x0_hr + h_hr, y0_hr: y0_hr + w_hr]
        else:
            # ---- 闂傚倷绀侀幉锟犮€冮崱妞曟椽寮介‖锛勬嚀铻ｉ柤濮愬€愰弸鏍煟鎼搭垳绉甸柛鐘崇墪閳诲秴顫濋懜鐢靛幐閻庡箍鍎卞ú锕傛倿閸ф鐓熸い鎾跺枎閸斻倖銇勯弴妯哄姦鐎规洜鍠栧畷婊勬媴娓氼垱袩闂傚倷绀侀幉锟犫€﹂崶顒€绐楅幖鎼厜缂嶆牠鏌熼幍顔碱暭闁绘帒顭烽弻锟犲礋椤愩倗顔婇柣銏╁灠濡繈寮?---
            if self.inp_size is None:
                h_lr = math.floor(img.shape[-2] / s + 1e-9)
                w_lr = math.floor(img.shape[-1] / s + 1e-9)
                img = img[:, :round(h_lr * s), :round(w_lr * s)]
                img_down = resize_fn(img, (h_lr, w_lr))
                crop_lr, crop_hr = img_down, img
            else:
                h_lr = math.floor(self.inp_size / s + 1e-9)
                w_lr = math.floor(self.inp_size / s + 1e-9)
                w_hr = round(w_lr * s)
                h_hr = round(h_lr * s)
                x0 = random.randint(0, img.shape[-2] - h_hr)
                y0 = random.randint(0, img.shape[-1] - w_hr)
                crop_hr = img[:, x0: x0 + h_hr, y0: y0 + w_hr]
                crop_lr = resize_fn(crop_hr, (h_lr, w_lr))

        # Optionally add noise to LR patches only.
        if self.add_noise and self.noise_sigma > 0:
            crop_lr = _add_rician_noise_to_magnitude(crop_lr, self.noise_sigma, self.noise_k_factor,
                                                     mode=self.noise_mode, foreground_only=self.foreground_only,
                                                     fg_threshold=self.fg_threshold)
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            if has_mask:
                crop_mask = augment(crop_mask)

        result = {
            'inp': crop_lr,
            'gt': crop_hr,
            'scale': s,
        }
        if has_mask:
            result['mask'] = crop_mask   # [1, H_hr, W_hr], integer labels stored as float32
        return result

@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('sr-implicit-paired-multiscale')
class SRImplicitPairedMultiScale(Dataset):

    def __init__(self, dataset, gt_size=None, scale_list=None, scale_probs=None,
                 scale_min=None, scale_max=None, augment=False, mode='train',
                 expand_for_eval=False, use_mask=True):
        self.dataset = dataset
        self.gt_size = gt_size
        self.augment = augment
        self.mode = mode
        self.expand_for_eval = expand_for_eval
        self.use_mask = use_mask

        if scale_list is not None and len(scale_list) > 0:
            self.scale_list = [float(s) for s in scale_list]
        elif scale_min is not None and scale_max is not None:
            self.scale_list = None
            self.scale_min = float(scale_min)
            self.scale_max = float(scale_max)
        else:
            raise ValueError('Provide either scale_list or both scale_min and scale_max.')

        if scale_probs is not None:
            probs = np.asarray(scale_probs, dtype=np.float64)
            if self.scale_list is None or len(probs) != len(self.scale_list):
                raise ValueError('scale_probs must match scale_list length.')
            probs = probs / probs.sum()
            self.scale_probs = probs.tolist()
        else:
            self.scale_probs = None

    def __len__(self):
        if self.mode != 'train' and self.expand_for_eval and self.scale_list is not None:
            return len(self.dataset) * len(self.scale_list)
        return len(self.dataset)

    def _resolve_raw(self, idx):
        raw = self.dataset[idx]
        if isinstance(raw, (tuple, list)):
            if len(raw) == 2:
                img_hr, img_mask = raw
            elif len(raw) == 3:
                _, img_hr, img_mask = raw
            else:
                raise ValueError(
                    f'sr-implicit-paired-multiscale expects hr-only, (hr, mask), or (lr, hr, mask); got len={len(raw)}'
                )
        else:
            img_hr = raw
            img_mask = None
        if img_mask is not None and img_mask.ndim == 2:
            img_mask = img_mask.unsqueeze(0)
        return img_hr, img_mask

    def _choose_scale(self, idx):
        if self.mode == 'train':
            if self.scale_list is not None:
                if self.scale_probs is None:
                    return float(random.choice(self.scale_list))
                return float(np.random.choice(self.scale_list, p=self.scale_probs))
            return float(np.random.uniform(self.scale_min, self.scale_max))

        if self.scale_list is not None:
            if self.expand_for_eval:
                return float(self.scale_list[idx % len(self.scale_list)])
            return float(self.scale_list[0])
        return float((self.scale_min + self.scale_max) / 2.0)

    def _base_index(self, idx):
        if self.mode != 'train' and self.expand_for_eval and self.scale_list is not None:
            return idx // len(self.scale_list)
        return idx

    def _crop_hr(self, img_hr, img_mask, crop_h, crop_w):
        h_hr, w_hr = img_hr.shape[-2:]
        if crop_h > h_hr or crop_w > w_hr:
            raise ValueError(
                f'gt_size {crop_h}x{crop_w} is larger than sample size {h_hr}x{w_hr}'
            )

        if self.mode == 'train':
            x0 = random.randint(0, h_hr - crop_h) if h_hr > crop_h else 0
            y0 = random.randint(0, w_hr - crop_w) if w_hr > crop_w else 0
        else:
            x0 = max((h_hr - crop_h) // 2, 0)
            y0 = max((w_hr - crop_w) // 2, 0)

        crop_hr = img_hr[:, x0:x0 + crop_h, y0:y0 + crop_w]
        crop_mask = None
        if img_mask is not None:
            crop_mask = img_mask[:, x0:x0 + crop_h, y0:y0 + crop_w]
        return crop_hr, crop_mask

    def __getitem__(self, idx):
        base_idx = self._base_index(idx)
        img_hr, img_mask = self._resolve_raw(base_idx)
        scale = self._choose_scale(idx)

        if self.gt_size is None:
            crop_hr = img_hr
            crop_mask = img_mask
        else:
            crop_hr, crop_mask = self._crop_hr(img_hr, img_mask, self.gt_size, self.gt_size)

        lr_h = max(1, int(round(crop_hr.shape[-2] / scale)))
        lr_w = max(1, int(round(crop_hr.shape[-1] / scale)))
        crop_lr = resize_fn(crop_hr, (lr_h, lr_w))

        # Align GT/mask to the model's round-trip output size so arbitrary
        # scales do not produce off-by-one mismatches during loss/validation.
        target_hr_h = max(1, int(round(lr_h * scale)))
        target_hr_w = max(1, int(round(lr_w * scale)))
        if crop_hr.shape[-2:] != (target_hr_h, target_hr_w):
            crop_hr = resize_fn(crop_hr, (target_hr_h, target_hr_w))
            if crop_mask is not None:
                crop_mask = F.interpolate(
                    crop_mask.unsqueeze(0).to(torch.float32),
                    size=(target_hr_h, target_hr_w),
                    mode='nearest',
                ).squeeze(0).to(crop_mask.dtype)

        if self.augment and self.mode == 'train':
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def _augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = _augment(crop_lr)
            crop_hr = _augment(crop_hr)
            if crop_mask is not None:
                crop_mask = _augment(crop_mask)

        result = {
            'inp': crop_lr.contiguous(),
            'gt': crop_hr.contiguous(),
            'scale': torch.tensor(scale, dtype=torch.float32).contiguous(),
        }
        if self.use_mask and crop_mask is not None:
            result['mask'] = crop_mask.contiguous()
        return result


def resize_fn(img, size):
    """Stable resize helper for MRI tensors/arrays/images, returning CHW float32."""
    if isinstance(size, int):
        size = (size, size)
    target_h, target_w = int(size[0]), int(size[1])

    if isinstance(img, Image.Image):
        tensor = transforms.ToTensor()(img)
    elif isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3):
                tensor = torch.from_numpy(arr)
            else:
                tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)))
        else:
            raise ValueError(f'Unsupported numpy shape for resize_fn: {arr.shape}')
    elif isinstance(img, torch.Tensor):
        tensor = img.detach().clone()
    else:
        tensor = transforms.ToTensor()(img)

    tensor = tensor.to(dtype=torch.float32)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 3:
        raise ValueError(f'Expected HW or CHW input in resize_fn, got shape {tuple(tensor.shape)}')

    return F.interpolate(
        tensor.unsqueeze(0),
        size=(target_h, target_w),
        mode='bicubic',
        align_corners=False,
    ).squeeze(0)
