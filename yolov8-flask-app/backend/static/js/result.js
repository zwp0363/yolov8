// 获取URL中的检测ID
const urlParams = new URLSearchParams(window.location.search);
const id = urlParams.get('id') || window.location.pathname.split('/').pop();

// 处理模型路径，只显示最后三个部分
function processModelPath(modelPath) {
    if (!modelPath) return 'N/A';
    // 统一使用Windows风格的反斜杠
    modelPath = modelPath.replace(/\/|\\/g, '\\');
    // 分割路径并取最后三个部分
    const parts = modelPath.split('\\').filter(Boolean); // 过滤空字符串
    const lastThreeParts = parts.slice(-3);
    return lastThreeParts.join('\\'); // 显示为Windows风格的反斜杠
}

async function loadResult() {
    try {
        const response = await fetch(`/api/result/${id}`);
        
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
        let detectionsHtml = `<p class="mb-2"><strong>检测时间:</strong> ${result.timestamp}</p>`;
        detectionsHtml += `<p class="mb-2"><strong>检测耗时:</strong> ${(result.elapsed_time * 1000).toFixed(2)} 毫秒</p>`;
        detectionsHtml += `<p class="mb-2"><strong>模型权重:</strong> ${processModelPath(result.model_path)}</p>`;
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
        console.error('加载结果时发生错误:', error);
        alert('加载结果时发生错误，请查看控制台获取详细信息');
    }
}

// 页面加载时加载结果
window.onload = loadResult;