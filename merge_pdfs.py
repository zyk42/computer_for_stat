import glob
import os
import sys

# 终极兼容：适配所有 pypdf/PyPDF2 版本（1.x/2.x/3.x）
def get_pdf_merger_class():
    try:
        # 优先导入新版 pypdf/PyPDF2 的 PdfMerger（2.0+/3.0+）
        try:
            from pypdf import PdfMerger
            return PdfMerger
        except ImportError:
            from PyPDF2 import PdfMerger
            return PdfMerger
    except ImportError:
        # 兼容极旧版本 PyPDF2 的 PdfFileMerger（1.x）
        try:
            from PyPDF2 import PdfFileMerger
            return PdfFileMerger
        except ImportError:
            raise ImportError(
                "请安装 pypdf 或 PyPDF2：\n"
                "pip install --upgrade pypdf\n"
                "或\n"
                "pip install PyPDF2<3.0.0"
            )

# 获取适配的 Merger 类
PdfMerger = get_pdf_merger_class()


def pick_six_pdfs():
    # 优先找 chapter_*.pdf 命名的文件
    files = sorted([f for f in glob.glob("chapter_*.pdf") if os.path.isfile(f)])
    if len(files) == 6:
        return files
    
    # 找不到6个则找所有pdf（排除合并后的文件）
    all_pdfs = sorted([f for f in glob.glob("*.pdf") if os.path.isfile(f)])
    exclude_files = {"book_merged.pdf", "merged.pdf"}
    all_pdfs = [f for f in all_pdfs if os.path.basename(f).lower() not in exclude_files]
    
    if len(all_pdfs) >= 6:
        return all_pdfs[:6]
    sys.exit(f"当前目录仅发现 {len(all_pdfs)} 个 PDF，无法合并为 6 个。")


def main():
    try:
        inputs = pick_six_pdfs()
        out = "book_merged.pdf"
        merger = PdfMerger()  # 直接用适配后的类
        
        # 逐个添加PDF文件，增加异常提示
        for idx, pdf in enumerate(inputs, 1):
            try:
                merger.append(pdf)
                print(f"已添加第 {idx} 个文件：{pdf}")
            except Exception as e:
                sys.exit(f"添加文件 {pdf} 失败：{str(e)}")
        
        # 写入合并后的文件
        with open(out, "wb") as f:
            merger.write(f)
        merger.close()
        
        print("\n✅ 合并完成！")
        print("📋 合并的文件列表：")
        for i, f in enumerate(inputs, 1):
            print(f" - {i}. {f}")
        print(f"📄 输出文件：{out}")
        
    except ImportError as e:
        sys.exit(f"❌ 依赖安装失败：{str(e)}")
    except Exception as e:
        sys.exit(f"❌ 程序执行失败：{str(e)}")


if __name__ == "__main__":
    main()