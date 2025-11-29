import os
from flask import Flask, request, jsonify, render_template, current_app, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import datetime
from google.cloud import storage # <--- 變更：匯入 GCS
from threading import Thread
import requests
import json
import markdown
from analysis_model import get_gemini_report_from_video
import uuid

# ====  設定 Flask 應用程式 ====
app = Flask(__name__)

# =======   Gemini API 金鑰   =======
# 警告：更安全的方式是將它設為環境變數
# os.environ.get('GEMINI_API_KEY')
GEMINI_API_KEY = "" # 貼上自己的金鑰 

# --- 2. 設定儲存和資料庫 ---
# --- 設定 GCS Bucket 名稱 ---
GCS_BUCKET_NAME = ''  # 請改成建立的 Bucket 名稱

# --- 初始化 GCS 客戶端 ---
# 程式會自動使用您設定的 GOOGLE_APPLICATION_CREDENTIALS 環境變數
storage_client = storage.Client()

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)

# --- 3. 建立資料庫模型 (Model) ---
class VideoUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300), nullable=False) # 存 UUID 檔名 (給系統用)
    
    display_name = db.Column(db.String(300), nullable=False) # 存原始檔名 (給人看)
    
    filepath = db.Column(db.String(500), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    status = db.Column(db.String(50), default='uploaded') 
    report_json = db.Column(db.Text, nullable=True)

def run_ai_in_background(app, video_id, public_url, side):
    """
    在「背景」執行的函式
    """
    print(f"[AI 背景任務]：開始分析 {public_url}")
    final_report_text = None
    try:
        # ===== 關鍵呼叫 (有 3 個參數) =====
        final_report_text = get_gemini_report_from_video(
            public_url, 
            side, # <--- 使用從瀏覽器傳來的 'side'
            GEMINI_API_KEY
        )
        print(f"[AI 背景任務]：AI 流程全部完成！")
        # ==================================
    except Exception as e:
        print(f"[AI 背景任務]：AI 流程失敗 {e}")
        final_report_text = f"AI 分析過程中發生錯誤: {e}"

    # ... (更新資料庫) ...
    with app.app_context():
        try:
            video = db.session.get(VideoUpload, video_id)
            if video:
                if final_report_text:
                    video.status = 'COMPLETED'
                    video.report_json = final_report_text # 儲存 Gemini 的文字
                else:
                    video.status = 'FAILED'
                
                db.session.commit()
                print(f"DB updated for video {video_id}")
        except Exception as e:
            print(f"DB update failed: {e}")
            db.session.rollback()

# --- 4. 建立 API 路由 (Endpoint) ---

@app.route('/api/upload', methods=['POST'])
def upload_video():
    #檢查檔案是否存在
    if 'videoFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['videoFile']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 讀取使用者選擇的 'side' (慣用手)
    # (來自 main.js 的 formData)
    side = request.form.get('side', 'right') # 預設為 'right'
    
    if file:
        # 確保檔案名稱安全
        original_filename = secure_filename(file.filename)

        #生成唯一檔名 (要存到 GCS 的)
        unique_filename = f"{uuid.uuid4().hex[:8]}_{original_filename}"
        
        try:
            #取得 GCS 儲存桶
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            
            #定義 'blob' 變數
            blob = bucket.blob(original_filename)
            
            #從串流上傳檔案到 GCS
            blob.upload_from_file(
                file.stream,
                content_type=file.content_type
            )
            
            # 將檔案設為公開 (以便 AI 分析和前端播放)
            # (確保GCS 儲存桶已設為「精細 (Fine-grained)」)
            blob.make_public()
            
            # 取得檔案的公開 URL
            public_url = blob.public_url

            # 存入資料庫
            new_video = VideoUpload(
                filename=unique_filename,      # 存 UUID 檔名 (用來刪除檔案)
                display_name=original_filename, # 存 原始檔名 (用來顯示)
                filepath=public_url,
                status='PROCESSING'
            )
            db.session.add(new_video)
            db.session.commit()
            
            # 取得新影片的 ID
            video_id = new_video.id 
            
            # 啟動「背景執行緒」來執行繁重的 AI 分析
            thread = Thread(
                target=run_ai_in_background, 
                args=(
                    current_app._get_current_object(), 
                    video_id, 
                    public_url, 
                    side  # 將 'side' 傳入背景任務
                )
            )
            thread.start() # 立即啟動，此函式會繼續往下執行
            
            # 「立刻」回傳成功訊息給瀏覽器
            return jsonify({
                'message': 'File uploaded, analysis started in background.',
                'video_id': video_id,
                'gcs_url': public_url
            }), 201 # 201 Created
            
        except Exception as e:
            # 如果 'try' 區塊中任何地方出錯 (包含 'blob' is not defined)
            db.session.rollback() # 回滾資料庫變更
            
            # 印出詳細錯誤到您的伺服器終端機 (方便除錯)
            print(f"Error during upload: {e}") 
            
            # 回傳錯誤訊息給瀏覽器
            return jsonify({'error': f'Cloud Storage or DB error: {str(e)}'}), 500

# --- 5. 建立前端頁面路由 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report/<int:video_id>')
def view_report(video_id):
    """
    顯示單一影片分析報告的頁面
    """
    try:
        video = db.session.get(VideoUpload, video_id)

        if not video:
            return "Report not found", 404
        
        # 建立一個變數來存放轉換後的 HTML
        report_html = None 
        
        # 檢查是否有報告文字，且狀態為 COMPLETED
        if video.status == 'COMPLETED' and video.report_json:
            # 在「後端」將 Markdown 文字轉換為 HTML
            report_html = markdown.markdown(video.report_json)
            
        elif video.status == 'FAILED' and video.report_json:
             # 如果失敗了，也轉換錯誤訊息 (它也可能是 Markdown)
            report_html = markdown.markdown(f"<span class='text-red-500'>{video.report_json}</span>")

        # 將 video 物件 和 轉換好的 report_html 一起傳給樣板
        return render_template('report.html', video=video, report_html=report_html)
        
    except Exception as e:
        return f"Error loading report: {str(e)}", 500

@app.route('/library')
def library():
    try:
        # 獲取當前頁碼，預設為第 1 頁，且必須是整數
        page = request.args.get('page', 1, type=int)
        
        # 獲取搜尋字詞
        search_term = request.args.get('search', '')

        # 建立基礎查詢
        query = VideoUpload.query.order_by(VideoUpload.upload_time.desc())

        if search_term:
            query = query.filter(VideoUpload.display_name.ilike(f"%{search_term}%"))
        
        # 執行分頁查詢
        # per_page=9: 每頁顯示 9 部影片
        # error_out=False: 如果頁碼超出範圍，不報錯，回傳空列表
        pagination = query.paginate(page=page, per_page=9, error_out=False)
        
        # 將 pagination 物件傳回前端
        return render_template('library.html', pagination=pagination, search_term=search_term)
        
    except Exception as e:
        return f"Error loading library: {str(e)}", 500

@app.route('/video/delete/<int:video_id>', methods=['POST'])
def delete_video(video_id):
    """
    刪除一個影片 (必須使用 POST 請求)
    """
    try:
        # 從資料庫找到這筆紀錄
        video = db.session.get(VideoUpload, video_id)
        
        if not video:
            return "Video not found", 404

        # 從 Google Cloud Storage 刪除實體檔案
        # (使用儲存在 DB 中的 'filename' 作為 GCS 上的 blob 名稱)
        try:
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(video.filename)
            blob.delete()
            print(f"Successfully deleted GCS file: {video.filename}")
        except Exception as e:
            # 如果 GCS 檔案已不存在，只印出日誌，但繼續刪除 DB 紀錄
            print(f"Warning: Could not delete GCS file {video.filename}. Error: {e}")

        # 從資料庫刪除這筆紀錄
        db.session.delete(video)
        db.session.commit()
        
        print(f"Successfully deleted DB record for ID: {video_id}")

        # 將使用者重新導向回影片庫
        return redirect(url_for('library'))

    except Exception as e:
        db.session.rollback()
        return f"Error deleting video: {str(e)}", 500

# --- 6. 啟動伺服器 ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)