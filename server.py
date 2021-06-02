from aiohttp import web
import socketio
from app import Application


sio = socketio.AsyncServer(
    async_mode='aiohttp',
    cors_allowed_origins='*'
)
app = web.Application()
cal = Application()
sio.attach(app)


async def index(request):
    with open('./public/index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')


@sio.event
def connect(sid, environ):
    print('connect ', sid)


@sio.event
async def image(sid, data):
    print('Data received from ', sid)
    y = cal.single_run(data)
    await sio.emit('result', y)


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


# app.router.add_static('/js', './public/js')
# app.router.add_static('/css', './public/css')
app.router.add_get('/', index)


if __name__ == '__main__':
    web.run_app(app)
