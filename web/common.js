const API_BASE = '/api';

// ========== 通用普通请求 ==========
async function request(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
        body: options.body ? JSON.stringify(options.body) : undefined
    });
    const result = await res.json();
    if (result.code !== 0) {
        throw new Error(result.message || '请求失败');
    }
    return result.data;
}

// ========== 通用SSE流式请求 ==========
function streamRequest(path, body, callbacks) {
    const { onContent, onSources, onDone, onError } = callbacks;
    let buffer = '';

    fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    }).then(response => {
        if (!response.ok) {
            throw new Error(`请求失败: ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');

        function processEvent(line) {
            if (!line.startsWith('data: ')) return;
            try {
                const event = JSON.parse(line.slice(6));
                switch (event.type) {
                    case 'content':
                        onContent && onContent(event.data);
                        break;
                    case 'sources':
                        onSources && onSources(event.data);
                        break;
                    case 'done':
                        onDone && onDone();
                        break;
                    case 'error':
                        onError && onError(event.message);
                        break;
                }
            } catch (e) {
            }
        }

        function read() {
            reader.read().then(({ done, value }) => {
                if (done) return;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop();
                lines.forEach(line => processEvent(line.trim()));
                read();
            }).catch(err => {
                onError && onError(err.message);
            });
        }
        read();
    }).catch(err => {
        onError && onError(err.message);
    });
}