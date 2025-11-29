// 確保 DOM (HTML) 完全載入後才執行
document.addEventListener('DOMContentLoaded', () => {

    // 獲取所有需要操作的 HTML 元素
    const uploadArea = document.getElementById('upload-area');
    const videoInput = document.getElementById('video-input'); // 隱藏的 input
    const selectFileBtn = document.getElementById('select-file-btn'); // 按鈕
    const progressArea = document.getElementById('progress-area'); // 上傳進度條
    const reportButtonArea = document.getElementById('report-button-area'); // 觀看報告按鈕

    // 1. 獲取新的「View Report」按鈕
    const viewReportBtn = document.getElementById('view-report-btn');

    // ===== ⬇️ 新增：獲取下拉選單 ⬇️ =====
    const pitcherSideSelect = document.getElementById('pitcher-side');

    // 2. 建立一個變數來儲存上傳成功的影片 ID
    let uploadedVideoId = null;

    // 當點擊 "Select a File" 按鈕時...
    selectFileBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // 防止事件冒泡到父層 (uploadArea)
        videoInput.click(); // 觸發隱藏的 input 點擊事件
    });

    // 當點擊整個上傳區域時...
    uploadArea.addEventListener('click', () => {
        videoInput.click(); // 也觸發隱藏的 input 點擊事件
    });

    // 當使用者在 input 中選擇了檔案後...
    videoInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            console.log('File selected:', file.name);
            // 執行上傳
            uploadFile(file);
        }
    });

    // 處理檔案上傳的非同步函式
    async function uploadFile(file) {
        // 1. 切換 UI：隱藏上傳區塊，顯示進度條
        uploadArea.style.display = 'none';
        progressArea.style.display = 'flex';

        // 2. 建立 FormData 物件來打包檔案
        const formData = new FormData();

        // 讀取下拉選單的當前值 ("left" 或 "right")
        const selectedSide = pitcherSideSelect.value; 
        
        // 將「檔案」和「慣用手」都加入到 FormData 中
        // 'videoFile' 這個 key 必須和 app.py 中
        // request.files['videoFile'] 的 key 一致
        formData.append('videoFile', file);
        formData.append('side', selectedSide); //


        try {
            // 3. 使用 fetch API 將檔案 POST 到後端 /api/upload
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
                // 注意：上傳 FormData 時，瀏覽器會自動設定 Content-Type，
                // 千萬不要手動設定 'Content-Type': 'multipart/form-data'
            });

            // 4. 等待並解析後端的回應 (JSON)
            const data = await response.json();

            if (response.ok) {
                // 5a. 上傳成功
                console.log('Upload successful:', data);

                // 3. 儲存後端傳回的 video_id
                uploadedVideoId = data.video_id;
                
                // 這裡模擬 AI 分析時間 (2秒)，然後顯示報告按鈕
                // 在真實世界中，您會用另一種方式 (如輪詢) 來檢查狀態
                setTimeout(() => {
                    progressArea.style.display = 'none';
                    reportButtonArea.style.display = 'flex';
                }, 2000); 

            } else {
                // 5b. 上傳失敗 (後端回傳錯誤)
                console.error('Upload failed:', data.error);
                alert('Upload failed: ' + data.error);
                // 重設 UI 讓使用者可以重試
                uploadArea.style.display = 'flex';
                progressArea.style.display = 'none';
            }
        } catch (error) {
            // 5c. 網路或其他錯誤
            console.error('Network or fetch error:', error);
            alert('An error occurred. Please check your connection and try again.');
            // 重設 UI 讓使用者可以重試
            uploadArea.style.display = 'flex';
            progressArea.style.display = 'none';
        }

        // 清空 input 的值，這樣使用者才能重新上傳同一個檔案
        videoInput.value = '';
    }

    // 4. 為「View Report」按鈕新增點擊事件監聽
    viewReportBtn.addEventListener('click', () => {
        if (uploadedVideoId) {
            // 如果我們有儲存的 ID，就跳轉到對應的報告頁面
            console.log('Jumping to report for video ID:', uploadedVideoId);
            window.location.href = `/report/${uploadedVideoId}`;
        } else {
            // 這不應該發生，但以防萬一
            console.error('No video ID found. Cannot view report.');
            alert('Could not find report ID. Please try uploading again.');
        }
    });

});