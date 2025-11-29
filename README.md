# AI Pitching Analysis System (Developer Guide)

這是我們「AI 投球教練」專案的開發者文件。請按照以下步驟在你的本機電腦上建置開發環境。

## 事前準備 (Prerequisites)

在開始之前，請確保你的電腦已安裝：

* **Python 3.8 或以上版本**: [下載 Python](https://www.python.org/downloads/)
* **Git**: [下載 Git](https://git-scm.com/)
* **Visual Studio Code** (推薦使用的編輯器)

---

## 安裝步驟 (Installation)

### 1. Clone 專案
打開終端機 (Terminal) 或 VS Code，執行以下指令將專案下載到本地：

```bash
git clone [https://github.com/s1311332041/ai-pitching-coach.git](https://github.com/s1311332041/ai-pitching-coach.git)
cd ai-pitching-coach
```
### 2. 建立虛擬環境 (Virtual Environment)

```bash
# 建立環境
python -m venv venv

# 啟用環境 (請在 cmd 執行，若用 PowerShell 請參閱下方疑難排解)
venv\Scripts\activate
```
注意： 啟用成功後，你的終端機前面會出現 (venv) 字樣。接下來的所有指令都要在 (venv) 狀態下執行。

### 3. 安裝依賴套件

```bash
pip install -r requirements.txt
```
