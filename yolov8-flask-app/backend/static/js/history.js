async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const history = await response.json();
        
        // 显示历史记录
        const historyList = document.getElementById('history-list');
        
        if (history.length === 0) {
            historyList.innerHTML = '<p class="text-center">暂无检测记录</p>';
            return;
        }
        
        let historyHtml = `
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>检测时间</th>
                            <th>检测耗时</th>
                            <th>检测结果</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        history.forEach(item => {
            const detectionCount = item.detections.length;
            const firstFewDetections = item.detections.slice(0, 3).map(d => d.class).join(', ');
            const moreDetections = detectionCount > 3 ? `...等${detectionCount}个目标` : '';
            const elapsedTime = item.elapsed_time ? (item.elapsed_time * 1000).toFixed(2) + ' 毫秒' : 'N/A';
            
            historyHtml += `
                <tr>
                    <td>${item.id}</td>
                    <td>${item.timestamp}</td>
                    <td>${elapsedTime}</td>
                    <td>${firstFewDetections} ${moreDetections}</td>
                    <td>
                        <a href="/result/${item.id}" class="btn btn-sm btn-primary me-2">查看</a>
                        <button class="btn btn-sm btn-danger" onclick="deleteRecord(${item.id})">删除</button>
                    </td>
                </tr>
            `;
        });
        
        historyHtml += `
                    </tbody>
                </table>
            </div>
        `;
        
        historyList.innerHTML = historyHtml;
    } catch (error) {
        console.error('加载历史记录时发生错误:', error);
        alert('加载历史记录时发生错误，请查看控制台获取详细信息');
    }
}

async function deleteRecord(id) {
    if (confirm('确定要删除这条检测记录吗？')) {
        try {
            const response = await fetch(`/api/delete/${id}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            if (result.success) {
                // 重新加载历史记录
                loadHistory();
            } else {
                alert(result.message || '删除失败');
            }
        } catch (error) {
            console.error('删除记录时发生错误:', error);
            alert('删除记录时发生错误，请查看控制台获取详细信息');
        }
    }
}

// 页面加载时加载历史记录
window.onload = loadHistory;