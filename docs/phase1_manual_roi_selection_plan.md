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
## 5. 已知 Smell（2026-04-24）

- Smell 1（manualRoiActive 早清零）根因：`setInitialTarget -> refreshManualTemplateFromLiveFrame -> setTemplateImages` 过程中会触发 `resetTracking(trigger=template_changed)`，其 `overlayView.post { reset() }` 异步回调在 `MANUAL_ROI state=active` 之后执行，导致出现 `state=clear reason=overlay_reset_reset_tracking` 的时间反转现象。
- 当前决策：`manualRoiActive` 仅作为一次性 edge trigger（用于手动圈选当帧放行），不作为帧级 level 信号；后续若新增帧级门控，必须使用独立状态位，禁止复用本 flag。
- Smell 2（越界/磁盘回退）已纳入 T4 防御：手动圈选路径新增 `bbox_clamped` 可观测字段，且当活体模板初始化失败时固定上报 `reason=fallback_forbidden`，不再回退磁盘模板偷跑锁定。
