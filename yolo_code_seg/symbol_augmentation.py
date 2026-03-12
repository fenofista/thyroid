"""
Symbol Augmentation - 移除超音波影像上的標記符號
可以直接整合到動態增強系統
"""
import cv2
import numpy as np
import random
from pathlib import Path
from PIL import Image


class SymbolRemover:
    """
    超音波影像符號移除器
    用於訓練時隨機移除標記符號，讓模型學習乾淨的影像
    """
    def __init__(self, symbol_folder, number_of_symbols=300, cache_symbols=True):
        """
        初始化符號移除器

        Args:
            symbol_folder: 符號圖片資料夾路徑
            number_of_symbols: 載入符號的數量
            cache_symbols: 是否快取符號（建議 True）
        """
        self.symbol_folder = Path(symbol_folder)
        self.number_of_symbols = number_of_symbols
        self.symbols_arr = []
        self.symbols_dilated = []  # 預計算膨脹的符號
        self.cache_symbols = cache_symbols

        # 載入符號
        self._load_symbols()

    def _load_symbols(self):
        """載入所有符號圖片並預計算膨脹的 mask"""
        if not self.symbol_folder.exists():
            print(f"⚠ 警告: 找不到符號資料夾: {self.symbol_folder}")
            return

        loaded_count = 0
        kernel = np.ones((3, 3), np.uint8)  # 預先建立 kernel

        for i in range(self.number_of_symbols):
            try:
                path = self.symbol_folder / f"symbol_{i + 1}.png"
                if not path.exists():
                    continue

                symbol = Image.open(path)
                symbol = np.array(symbol)

                # 確保是灰階
                if len(symbol.shape) == 3:
                    symbol = cv2.cvtColor(symbol, cv2.COLOR_RGB2GRAY)

                self.symbols_arr.append(symbol)

                # 🚀 預計算膨脹的 mask（避免訓練時重複計算）
                dilated = cv2.dilate(symbol.astype(np.uint8), kernel, iterations=1)
                self.symbols_dilated.append(dilated != 0)  # 直接存 boolean mask

                loaded_count += 1

            except Exception as e:
                print(f"⚠ 載入符號失敗 {path}: {e}")
                continue

        if loaded_count > 0:
            print(f"✓ 成功載入 {loaded_count} 個符號（已預計算膨脹 mask）")
        else:
            print(f"⚠ 沒有載入任何符號")

    def remove_symbols_randomly(self, image, prob=1.0):
        """
        在影像上隨機移除符號（模擬清理標記）- 優化版本

        Args:
            image: 輸入影像 (H, W) 或 (H, W, C)
            num_symbols: 要移除的符號數量
            prob: 應用此增強的機率

        Returns:
            processed_image: 處理後的影像
        """
        # 機率判斷
        if random.random() > prob or len(self.symbols_arr) == 0:
            return image

        # 🚀 優化：直接在原始圖像上 inpaint（彩色或灰階）
        H, W = image.shape[:2]

        # 建立 inpaint mask
        inpaint_mask = np.zeros((H, W), dtype=np.uint8)

        # 🚀 優化：隨機選擇索引（避免重複選擇）
        num_available = len(self.symbols_arr)
        indices = random.choices(range(num_available), k=min(self.number_of_symbols, num_available))

        # 隨機貼上符號
        for idx in indices:
            mask = self.symbols_dilated[idx]  # 🚀 使用預計算的膨脹 mask
            h, w = mask.shape

            # 確保不會超出邊界
            if h >= H // 2 or w >= W:
                continue

            # 隨機位置（只在上半部，因為標記通常在這裡）
            y = np.random.randint(0, max(1, H // 2 - h))
            x = np.random.randint(0, max(1, W - w))

            # 🚀 優化：直接使用預計算的 boolean mask
            inpaint_mask[y:y+h, x:x+w][mask] = 255

        # 使用 inpaint 移除符號
        if np.any(inpaint_mask):
            # 🚀 優化：使用更快的 NS 算法，減小半徑
            inpainted_img = cv2.inpaint(
                image,
                inpaint_mask,
                inpaintRadius=2,  # 從 3 減到 2（更快）
                flags=cv2.INPAINT_NS  # NS 比 TELEA 快
            )
            return inpainted_img
        else:
            return image

    def __call__(self, image, prob=1.0):
        """讓 class 可以像 function 一樣呼叫"""
        return self.remove_symbols_randomly(image, prob)





# ============================================================================
# 使用範例
# ============================================================================

if __name__ == '__main__':
    # 方法 1: 使用 Class（推薦）
    print("="*60)
    print("方法 1: 使用 SymbolRemover Class")
    print("="*60)

    symbol_folder = "../data/CG_data/all_data_png/clean_symbols/gray"
    remover = SymbolRemover(symbol_folder, number_of_symbols=300)

    # 讀取測試圖片
    test_image_path = "../data/CG_data/all_data_png/nodule_not_clean_yolo/train/images"
    test_images = list(Path(test_image_path).glob("*.png"))

    if len(test_images) > 0:
        img_path = test_images[0]
        image = cv2.imread(str(img_path))

        # 應用符號移除
        processed = remover(image, prob=1.0)

        print(f"✓ 處理圖片: {img_path.name}")
        print(f"  原始尺寸: {image.shape}")
        print(f"  處理後尺寸: {processed.shape}")

        # 儲存結果
        output_path = Path("symbol_removal_test.png")
        cv2.imwrite(str(output_path), processed)
        print(f"✓ 結果已儲存: {output_path}")


