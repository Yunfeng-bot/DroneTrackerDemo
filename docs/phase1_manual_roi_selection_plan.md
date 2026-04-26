# Phase 1: 灞忓箷妗嗛€?(Manual ROI Selection) 钀藉湴璁″垝

## 1. 鐩爣涓庝环鍊?瑙ｅ喅 `template` (浜嬪厛鎷嶆憚) 涓?`real` (瀹炴櫙) 涔嬮棿鐨勭壒寰侀敊浣嶏紙Domain Gap锛夐棶棰樸€傞€氳繃璁╃敤鎴峰湪瀹為檯璧烽鐨勫睆骞曚笂褰撳満妗嗛€?ROI锛屽嵆鍒荤敓鎴?ORB 妯℃澘锛屼娇涓よ€呯殑鍏夌収銆佽瑙掋€佸昂搴﹂珮搴︿竴鑷达紝褰诲簳娑堥櫎妯℃澘閿欓厤閫犳垚鐨勯閿侀毦闂銆?
## 2. UI 浜や簰鏂规 (PreviewView)
- **瑙﹀彂鏈哄埗**锛氬湪 `PreviewView`锛堣棰戦瑙堝尯锛変笂澧炲姞闀挎寜锛圠ong Press锛夋墜鍔胯繘鍏?妗嗛€夋ā寮?銆?- **浜や簰鍔ㄤ綔**锛?  - 鎵嬫寚鎷栧姩褰㈡垚鐭╁舰閬僵銆?  - 鏉惧紑鎵嬫寚锛圓CTION_UP锛夌‘璁?ROI 鍖哄煙銆?  - 灞忓箷鎻愮ず锛?姝ｅ湪鎻愬彇鐩爣鐗瑰緛..."
- **瑙嗚鍙嶉**锛氱粯鍒跺疄鏃跺姩鎬佺煩褰㈣竟妗嗐€?
## 3. 浠ｇ爜鎸傝浇鐐逛笌淇敼娓呭崟 (Hook Points)
涓轰簡璁?Codex / 瀹炴柦鑰呰兘蹇€熷垏鍏ワ紝浠ヤ笅鏄唬鐮佷慨鏀归敋鐐癸細

- **`MainActivity.kt`**: 
  - 澧炲姞瀵?`overlayView` 鐨勮Е鎽哥洃鍚?(`OnTouchListener`)銆?  - 澶勭悊 `ACTION_DOWN` (璁板綍璧风偣)銆乣ACTION_MOVE` (鏇存柊鐢绘)銆乣ACTION_UP` (瑙﹀彂鎴彇)銆?- **`TrackingOverlayView.kt`**:
  - 鏂板 `drawSelectionRect(rect: Rect?)` 鏂规硶锛岀敤浜庡湪 Canvas 涓婄粯鍒剁敤鎴峰綋鍓嶆鍦ㄦ嫋鎷界殑妗嗐€?- **`OpenCVTrackerAnalyzer.kt`**:
  - 鏂板 API锛歚fun setTemplateFromLiveFrame(roi: Rect)`銆?  - 鍐呴儴瀹炵幇锛氬彇褰撳墠鏈€鏂颁竴甯?`Mat`锛屾墽琛?`mat.submat(roi)` 鎴彇 Patch銆?  - 灏嗘 Patch 浣滀负鍔ㄦ€佸熀鍑嗕紶鍏?`rebuildTemplatePyramid` 涓?`NativeTrackerBridge.init`锛屼粠鑰岃烦杩囧師鏈夌殑鏈湴鏂囦欢鍔犺浇閫昏緫銆?  - 鐘舵€佹祦杞細杩娇鐘舵€佹満杩涘叆 `TRACKING`锛屾垨杩涘叆涓存椂 `ACQUIRE` 绛夊緟绋冲畾銆?
## 4. 楠屾敹 SOP锛堟祴璇曠煩闃碉級
閽堝璇ユ満鍒剁殑鍙敤鎬ц竟鐣岋紙鍍忕礌涓嬮檺锛夛紝鎴戜滑璁惧畾浠ヤ笅楠屾敹娴嬭瘯鍙ｅ緞锛?
- **鎸囨爣 1锛氭渶澶у彲杩借窛绂?(灏哄害閫€鍖栨瀬闄?** 
  - **鎿嶄綔**锛氬湪 5 绫宠繎璺濈妗嗛€変竴涓竟闀?50cm 鐨勭洰鏍囷紝鐒跺悗鏃犱汉鏈虹洿绾垮悗閫€/鎷夐珮銆?  - **楠屾敹闃堝€?*锛氭祴閲忓苟璁板綍鐩爣鍦ㄧ敾闈㈠崰姣旈檷鑷冲灏戯紙鐧惧垎姣旀垨鍍忕礌鏁帮級鏃剁郴缁熷彂鐢熸柇閿併€?- **鎸囨爣 2锛氭渶灏忓彲杩界洰鏍囧昂瀵?(鍍忕礌涓嬮檺)**
  - **鎿嶄綔**锛氬湪 20 绫冲灏濊瘯鐩存帴妗嗛€変竴涓皬鐗╀綋銆?  - **楠屾敹闃堝€?*锛氱‘璁ゅ綋妗嗗唴鍍忕礌浣庤嚦浣曠绋嬪害鏃讹紝ORB 鏃犳硶鎻愬彇瓒冲鐗瑰緛鐐癸紙鎶涘嚭寮傚父鎴栬鍛婏級銆?- **鎸囨爣 3锛氭椂闂寸ǔ瀹氭€?(绋冲畾璺熻釜鐜?**
  - **鎿嶄綔**锛氶潤鎬佽窛绂诲畬鎴愭閫夊悗锛屽洿缁曠洰鏍囧钩绉汇€佹棆杞瑙掋€?  - **楠屾敹闃堝€?*锛?0s 鍐呯殑 `steady_track_window` (鍗?TRACKING 鐘舵€佸崰姣? 鈮?95%銆?
## 5. 已知 Smell（2026-04-24 → 2026-04-26 收敛）

### Smell 1（manualRoiActive 早清零）· 已解

- 根因：`setInitialTarget -> refreshManualTemplateFromLiveFrame -> setTemplateImages` 过程中会触发 `resetTracking(trigger=template_changed)`，其 `overlayView.post { reset() }` 异步回调在 `MANUAL_ROI state=active` 之后执行，导致出现 `state=clear reason=overlay_reset_reset_tracking` 的时间反转现象。
- 决策：`manualRoiActive` 仅作为一次性 edge trigger（用于手动圈选当帧放行），不作为帧级 level 信号；帧级门控由独立的 `manualRoiSessionActive` 承载（Batch 1 引入）。
- 验证：Batch 1~7 联动后 `MANUAL_ROI state=active` 与 `state=clear` 的事件序列与时间戳已对齐，无再发反转。

### Smell 2（越界/磁盘回退）· 已解

- 根因：手动圈选失败时回退到磁盘模板，违反 Phase 1 "用户的圈是 ground truth" 设计原则。
- 防御：手动圈选路径新增 `bbox_clamped` 可观测字段，活体模板初始化失败时固定上报 `reason=fallback_forbidden`，并通过 `clearManualRoiState("manual_init_failed")` 确保不再有偷跑锁定。

### Smell 3（NCC 时间稳定性误用）· 已解（Batch 11）

- 根因：Batch 9 把 NCC 引入 TRACKING 阶段周期检查，但用单帧阈值（0.55）做 veto 决策。NCC 是逐帧采样的连续噪声信号，NCNN tracker init 后头几帧内部状态未稳，常出现单帧 NCC 跌到 0.4~0.5；单帧 veto 立刻触发 `track_ncc_drift`，把刚锁定的目标误判漂移。
- 防御（Batch 11 三件套）：
  1. **Grace window**：`initializeTracker` 成功后 30 帧（~1s）内 NCC 只采样不 veto（`MANUAL_ROI_TRACK_NCC stage=grace`）
  2. **滑窗中位数**：保留最近 4 个 NCC 样本，median < 0.40 才判持续漂移
  3. **Panic 路径**：单帧 NCC < 0.20 立刻 veto（catch 灾难性误锁到完全异物）
- 验证（V3 真机 2026-04-26）：grace 期 NCC avg 0.922 / median 0.924；首锁后持续追踪 10.668s 不被踢；挥开后 panic 正确触发；回归后 fingerprint_pass relock_unblocked，二次追踪持续 18.152s 无丢锁。

### Smell 4（错跟相似物 + 渐进漂移）· 已解（Batch 6 + 7v2 + 8 + 10）

- 根因：目标移出画面后，NCNN 倾向锁到附近相似物体（同款空调/楼顶等）。早期纯 ORB 重锁路径（`good=25 inliers=22 conf=0.88` 这种"看起来够强"的候选）会被接受，造成视觉上"locks=1 lost=0 但跟在错物体上"。
- 防御（多层叠加）：
  1. TRACKING 阶段几何 veto（Batch 6）：anchor_drift / area_grow / area_shrink / frame_jump / edge_hug 五维硬阈值
  2. relock NCC 身份验证（Batch 7v2）：保存用户圈选时的 patch，重锁候选必须通过 NCC ≥ 0.70
  3. 堵 `orb_temporal_confirm` bypass（Batch 8）：所有手动会话内 `manual_roi_direct` / `orb_temporal_confirm` / `auto_lock_init` 路径统一走 `passesManualRoiRelockGate`
  4. NCC 算法修正（Batch 10）：`computeNccAgainstManualPatch` 改为 search-window matchTemplate（patch 跟随当前 box 缩放，在 ±20% 边界内取最大 NCC），消除等尺寸 NCC 对像素级对齐误差的过敏
- 验证：V3 场景 B（相似异物）连续 fingerprint_mismatch 拒绝，无错锁；V3 场景 A（原目标回归）NCC = 0.889 通过，自动重锁成功。

## 6. Phase 1 已知边界（不在 T4.2 修复范围）

以下问题已识别但定为 Phase 2 / 独立债，不阻塞 Phase 1 收尾：

- **Scale 大变化 / 视角剧变**：手机靠近/拉远目标超过 2x 时，patch 缩放伪影会拖低 NCC；当前 search-window 单尺度匹配的容差有限。改进方向：multi-scale matchTemplate 或 ORB descriptor 距离。
- **verify_realign in-vivo 验证缺失**：`maybeVerifyTracking` 返回 realign 决策的路径在 Phase 1 真机测试中从未自然触发；`initializeTracker` 入口的统一 NCC 守门已防御该路径，但缺少现场证据。改进方向：unit test 用合成 `TrackVerifyDecision` 验证。
- **L1 外场稳定性**：MVP-5 l1 回放 ~33% 概率落 slow path（first_lock 7.7~12s vs baseline 3.5s），不影响 Phase 1 手动圈选场景，独立登记为 L1 验收前债项。
