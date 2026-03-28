document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const imageFile = document.getElementById('image-upload').files[0];
    
    if (!imageFile) {
        alert('请选择一张图片');
        return;
    }
    
    formData.append('image', imageFile);
    
    // 显示上传的图像
    const uploadedImage = document.getElementById('uploaded-image');
    uploadedImage.innerHTML = `<img src="${URL.createObjectURL(imageFile)}" alt="上传的图像">`;
    
    try {
        // 发送请求到后端
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // 显示检测结果
        const resultImage = document.getElementById('result-image');
        const detections = document.getElementById('detections');
        
        // 将十六进制字符串转换为图像（浏览器兼容版本）
        function hexToBytes(hex) {
            const bytes = new Uint8Array(hex.length / 2);
            for (let i = 0; i < hex.length; i += 2) {
                bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
            }
            return bytes;
        }
        
        const imgBytes = hexToBytes(result.image);
        const blob = new Blob([imgBytes], { type: 'image/jpeg' });
        const imgUrl = URL.createObjectURL(blob);
        
        resultImage.innerHTML = `<img src="${imgUrl}" alt="检测结果">`;
        
        // 显示检测到的目标
        let detectionsHtml = '';
        result.detections.forEach((detection, index) => {
            detectionsHtml += `
                <div class="detection-item">
                    <h5>目标 ${index + 1}</h5>
                    <p>类别: ${detection.class}</p>
                    <p>置信度: ${(detection.confidence * 100).toFixed(2)}%</p>
                    <p>位置: (${detection.bbox[0].toFixed(2)}, ${detection.bbox[1].toFixed(2)}) - (${detection.bbox[2].toFixed(2)}, ${detection.bbox[3].toFixed(2)})</p>
                </div>
            `;
        });
        
        detections.innerHTML = detectionsHtml;
    } catch (error) {
        console.error('检测过程中发生错误:', error);
        alert('检测过程中发生错误，请查看控制台获取详细信息');
    }
});