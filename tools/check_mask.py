"""工业语义分割数据集（MASK 标注）全面合规性检测脚本。

检查功能与指标：
  1. 配对完整性检查 (Pairing Check)：原图与 Mask 是否 1:1 严格配对，检测孤立原图与孤立掩码。
  2. 损坏文件解码检查 (Corruption Check)：检测 OpenCV / PIL 是否能正常加载原图与掩码。
  3. 空间分辨率对齐检查 (Resolution Alignment)：校验原图 (H, W) 与 Mask (H, W) 是否完全一致。
  4. 像素通道格式检查 (Channel Format Check)：校验 Mask 是否为标准的单通道二值/灰度矩阵，对误存为 3/4 通道 RGB(A) 报警。
  5. 类别 ID / 像素数值校验 (Unique Pixel Check)：
     - 正常二分类 Mask 仅允许包含 0（背景）和 1（正样本）；
     - 若出现 255（常见二值图误作类别ID）给出严正警告与一键修复建议；
     - 若出现负数或除 [0, 1, 255] 外的未知编号，直接报错指出。
  6. 极小目标与全黑图形态统计 (Morphology & Area Stats)：统计全背景（零目标）图片数量及细微导线占比极低（<0.05%）的高风险图。

运行环境：推荐使用项目现有的 Python 环境（支持 tqdm, numpy, cv2 / PIL）。
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 全局配置参数
# ---------------------------------------------------------------------------
# 数据集根目录
DATA_ROOT = Path(__file__).resolve().parent / "Breakage_small_seg_1024x416"
# 允许的合法类别 ID 列表（0: 背景, 1: fpc）
LEGAL_CLASS_IDS = {0, 1}
# 允许的特殊忽略索引（如果有些数据集把边界标记为 255 忽略计算）
ALLOWED_IGNORE_INDEX = 255
# 极小目标面积警戒线（如目标像素在全图中占比低于 0.05% 触发提醒）
TINY_TARGET_RATIO_THRESHOLD = 0.0005


class MaskDatasetChecker:
    """Mask 数据集自动化健康检查器"""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.splits = ["train", "val"]
        self.errors = defaultdict(list)
        self.warnings = defaultdict(list)
        self.stats = defaultdict(lambda: defaultdict(int))
        self.area_ratios = defaultdict(list)

    def run(self):
        print("=" * 70)
        print(f"[{'工业语义分割数据集（MASK 标注）全面检查仪' : ^28}]")
        print(f"正在扫描数据集目录：{self.data_root}")
        print("=" * 70)

        if not self.data_root.exists():
            print(f"❌ 严重错误：找不到数据集根目录 {self.data_root} ！请检查路径是否正确。")
            return False

        for split in self.splits:
            self._check_split(split)

        self._print_report()
        return len(self.errors) == 0

    def _check_split(self, split: str):
        img_dir = self.data_root / "img_dir" / split
        ann_dir = self.data_root / "ann_dir" / split

        print(f"\n---> 开始检查 [{split.upper()}] 集...")
        if not img_dir.exists() or not ann_dir.exists():
            msg = f"目录缺失：img_dir={img_dir.exists()}, ann_dir={ann_dir.exists()}"
            self.errors[split].append(msg)
            print(f"❌ {msg}")
            return

        # 获取文件名列表（按主名称配对）
        img_files = {f.stem: f for f in img_dir.iterdir() if f.is_file()}
        ann_files = {f.stem: f for f in ann_dir.iterdir() if f.is_file()}

        # 1. 检查孤立原图和孤立 Mask
        orphan_imgs = set(img_files.keys()) - set(ann_files.keys())
        orphan_anns = set(ann_files.keys()) - set(img_files.keys())
        common_stems = set(img_files.keys()) & set(ann_files.keys())

        if orphan_imgs:
            self.errors[split].append(f"发现 {len(orphan_imgs)} 张原图缺少对应 Mask，示例: {list(orphan_imgs)[:3]}")
        if orphan_anns:
            self.errors[split].append(f"发现 {len(orphan_anns)} 个 Mask 缺少对应原图，示例: {list(orphan_anns)[:3]}")

        self.stats[split]["total_imgs"] = len(img_files)
        self.stats[split]["total_anns"] = len(ann_files)
        self.stats[split]["paired_count"] = len(common_stems)

        if len(common_stems) == 0:
            self.errors[split].append("当前划分下没有找到任何配对的图与标注！")
            return

        # 2. 逐对检查深度格式、分辨率对齐与像素值合规性
        has_255_warning_issued = False
        for stem in tqdm(sorted(common_stems), desc=f"扫描 {split} 数据集质量", unit="张"):
            img_path = img_files[stem]
            ann_path = ann_files[stem]

            # 加载原图与 Mask
            img = cv2.imread(str(img_path))
            # OpenCV imread 默认读取为 BGR，指定 IMREAD_UNCHANGED 保持原始通道和位深
            mask_raw = cv2.imread(str(ann_path), cv2.IMREAD_UNCHANGED)

            if img is None:
                self.errors[split].append(f"原图损坏无法解码: {img_path.name}")
                continue
            if mask_raw is None:
                self.errors[split].append(f"Mask损坏无法解码: {ann_path.name}")
                continue

            h_img, w_img = img.shape[:2]
            h_mask, w_mask = mask_raw.shape[:2]

            # (A) 尺寸对齐校验
            if (h_img, w_img) != (h_mask, w_mask):
                self.errors[split].append(
                    f"尺寸不匹配 [{stem}]: 原图({w_img}x{h_img}) vs Mask({w_mask}x{h_mask})"
                )

            # (B) 通道格式校验
            if mask_raw.ndim > 2:
                # 检查若为 3 或 4 通道，各个通道数值是否完全一致
                if mask_raw.ndim == 3 and mask_raw.shape[2] >= 3:
                    if not np.array_equal(mask_raw[:, :, 0], mask_raw[:, :, 1]) or not np.array_equal(
                        mask_raw[:, :, 0], mask_raw[:, :, 2]
                    ):
                        self.errors[split].append(f"Mask格式异常 [{stem}]: RGB通道数值不一致，可能是普通彩色图！")
                    else:
                        self.warnings[split].append(
                            f"Mask保存格式为 {mask_raw.shape[2]} 通道 [{stem}]，建议转为标准的单灰度/8位通道。"
                        )
                mask = mask_raw[:, :, 0]
            else:
                mask = mask_raw

            # (C) 像素值与类别 ID 合规性检查
            unique_vals = set(np.unique(mask))
            # 过滤合法 ID
            unknown_vals = unique_vals - LEGAL_CLASS_IDS

            if unknown_vals:
                # 如果包含 255
                if ALLOWED_IGNORE_INDEX in unknown_vals:
                    other_unknowns = unknown_vals - {ALLOWED_IGNORE_INDEX}
                    if not has_255_warning_issued:
                        self.warnings[split].append(
                            f"检测到 Mask 中包含像素值 255 (示例图: {stem}.png)。请确认 255 是被配置成了 ignore_index，"
                            "还是说这是普通二值图(0和255)被误当作了标注 ID！若把 255 当作类 1，训练将因数组越界直接崩溃！"
                        )
                        has_255_warning_issued = True
                    if other_unknowns:
                        self.errors[split].append(
                            f"非法的未定义类别 ID [{stem}]: 发现非标像素值 {other_unknowns}"
                        )
                else:
                    self.errors[split].append(
                        f"非法的类别 ID [{stem}]: 期望仅包含 {LEGAL_CLASS_IDS}，实际发现像素值 {unique_vals}"
                    )

            # (D) 目标形态统计与空图检测
            # 计算正类面积比例（假设 class 1 代表 fpc 正类，或者如果只标了 255 则把 255 算正类）
            pos_pixels = np.sum((mask == 1) | (mask == 255))
            total_pixels = h_mask * w_mask
            ratio = pos_pixels / total_pixels
            self.area_ratios[split].append(ratio)

            if pos_pixels == 0:
                self.stats[split]["empty_masks"] += 1
                # 显式把未标注（全黑全0）的 Mask 记入警告名单，并记录文件名
                self.warnings[split].append(f"【漏标/未标注】Mask 没有任何正样本目标(全是黑背景 0): {stem}.png")
            elif ratio < TINY_TARGET_RATIO_THRESHOLD:
                self.stats[split]["tiny_targets"] += 1
                self.warnings[split].append(f"【极细小目标】正样本占全图仅 {ratio*100:.4f}%: {stem}.png")

    def _print_report(self):
        print("\n" + "=" * 70)
        print(f"[{'数据自检诊断终极报告' : ^30}]")
        print("=" * 70)

        total_errs = sum(len(e) for e in self.errors.values())
        total_warns = sum(len(w) for w in self.warnings.values())

        # 将所有异常详细列表写入本地文件，方便一键定位
        report_log_path = self.data_root.parent / "mask_check_issues_list.txt"
        with open(report_log_path, "w", encoding="utf-8") as f_log:
            f_log.write("=" * 70 + "\n")
            f_log.write("         工业语义分割 MASK 标注自检详细诊断报告\n")
            f_log.write("=" * 70 + "\n\n")

            for split in self.splits:
                stats = self.stats[split]
                if stats["total_imgs"] == 0:
                    print(f"\n📂 [{split.upper()} 集] 目录为空或不存在。")
                    f_log.write(f"[{split.upper()} 集] 目录为空或不存在。\n")
                    continue

                ratios = self.area_ratios[split]
                avg_ratio = np.mean(ratios) * 100 if ratios else 0.0
                max_ratio = np.max(ratios) * 100 if ratios else 0.0
                min_pos_ratio = (
                    np.min([r for r in ratios if r > 0]) * 100 if any(r > 0 for r in ratios) else 0.0
                )

                print(f"\n📊 [{split.upper()} 集统计面板]")
                print(f"  • 原图总数        : {stats['total_imgs']} 张")
                print(f"  • 掩码总数        : {stats['total_anns']} 张")
                print(f"  • 配对成功数      : {stats['paired_count']} 对")
                print(f"  • 全黑空图 (毫无目标) : {stats['empty_masks']} 张 ({stats['empty_masks']/max(1,stats['paired_count'])*100:.1f}%)")
                print(f"  • 极细小缺陷 (<{TINY_TARGET_RATIO_THRESHOLD*100:.2f}%): {stats['tiny_targets']} 张")
                print(f"  • 正样本平均占比  : {avg_ratio:.2f}% (单图最大: {max_ratio:.2f}%, 正图最小: {min_pos_ratio:.4f}%)")

                f_log.write(f"\n--- [{split.upper()} 集统计] ---\n")
                f_log.write(f"原图总数: {stats['total_imgs']}, 掩码总数: {stats['total_anns']}, 配对数: {stats['paired_count']}\n")
                f_log.write(f"毫无目标的全黑/未标注掩码: {stats['empty_masks']} 张\n\n")

                # 打印并写入错误
                if self.errors[split]:
                    print(f"\n  ❌ [{split.upper()} 集发现 {len(self.errors[split])} 项严重错误]:")
                    f_log.write(f"[{split.upper()} 集 - 严重错误列表 ({len(self.errors[split])}项)]\n")
                    for err in self.errors[split]:
                        f_log.write(f"  [Error] {err}\n")
                    for err in self.errors[split][:15]:  # 控制终端输出前15个
                        print(f"     [!] {err}")
                    if len(self.errors[split]) > 15:
                        print(f"     ...等剩余 {len(self.errors[split])-15} 项错误已完整写入日志: {report_log_path}")

                # 打印并写入警告/未标注提示
                if self.warnings[split]:
                    print(f"\n  ⚠️ [{split.upper()} 集发现 {len(self.warnings[split])} 项潜在风险/漏标警示]:")
                    f_log.write(f"\n[{split.upper()} 集 - 风险/未标注/超小目标列表 ({len(self.warnings[split])}项)]\n")
                    for warn in self.warnings[split]:
                        f_log.write(f"  [Warning] {warn}\n")
                    # 优先展示【漏标/未标注】的提示
                    unmarked_warns = [w for w in self.warnings[split] if "【漏标" in w]
                    other_warns = [w for w in self.warnings[split] if "【漏标" not in w]
                    
                    if unmarked_warns:
                        print(f"     👉 【重点注意】发现以下 {len(unmarked_warns)} 张 Mask 完全没有任何标注目标（全是背景0）：")
                        for w in unmarked_warns[:15]:
                            print(f"        • {w}")
                        if len(unmarked_warns) > 15:
                            print(f"        ...其余 {len(unmarked_warns)-15} 张漏标图已完整保存到日志。")
                    
                    if other_warns:
                        for w in list(dict.fromkeys(other_warns))[:8]:
                            print(f"     [?] {w}")

        print("\n" + "-" * 70)
        print(f"📋 完整的单图排查明细（含所有漏标/异常文件名）已保存至: {report_log_path}")
        if total_errs == 0:
            print("🎉 恭喜！数据集文件完整度、尺寸匹配与类别 ID 校验通过，无致命报错，可直接用于模型训练！")
            if total_warns > 0:
                print("💡 请务必查看上方黄字提示以及日志，尤其是那些【漏标/未标注】的全黑掩码！")
        else:
            print(f"🛑 检查未通过！共计发现 {total_errs} 处严重错误，请根据导出的清单解决后继续训练！")
        print("=" * 70)


if __name__ == "__main__":
    checker = MaskDatasetChecker(DATA_ROOT)
    success = checker.run()
    sys.exit(0 if success else 1)
