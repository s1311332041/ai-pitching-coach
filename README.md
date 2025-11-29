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
    啟用成功後，你的終端機前面會出現 (venv) 字樣。接下來的所有指令都要在 (venv) 狀態下執行。

### 3. 安裝依賴套件

```bash
pip install -r requirements.txt
```
### 設定金鑰與模型 (Configuration)

1. Google Cloud Storage 金鑰
    檔案名稱： keyfile.json

    位置： 放在專案根目錄 (/ai-pitching-coach/keyfile.json)

    設定環境變數： 執行程式前，你需要設定環境變數指向這個檔案。

    Windows (CMD): set GOOGLE_APPLICATION_CREDENTIALS=keyfile.json

    Mac/Linux: export GOOGLE_APPLICATION_CREDENTIALS="keyfile.json"

2. Gemini API Key
    打開 app.py (或 analysis_model.py)。

    找到 GEMINI_API_KEY 變數。

    填入我們申請的 API Key (請勿將含 Key 的程式碼 Push 上去)。

3. MediaPipe 模型檔
    請建立一個 models 資料夾。

    將 pose_landmarker_full.task 等模型檔案放入 models/ 資料夾中。

### 啟動伺服器 (Running the App)
```bash
python app.py
```
    如果成功，會看到：Running on http://127.0.0.1:5000

    打開瀏覽器前往該網址即可開始測試。

專案結構說明
    app.py: 後端主程式 (路由、資料庫、背景任務)。

    analysis_model.py: AI 核心邏輯 (MediaPipe + Gemini)。

    templates/: HTML 網頁 (首頁、影片庫、報告頁)。

    static/: CSS 和 JavaScript 檔案。

    instance/database.db: 本地資料庫 (由系統自動生成)。