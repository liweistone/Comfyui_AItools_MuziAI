import asyncio
import threading
import json
import random
import aiohttp
import os
import time
import platform
from pathlib import Path
from tqdm import tqdm
import folder_paths
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode

# ====================== 跨平台通知支持 ======================
class Notifier:
    @staticmethod
    def notify(message):
        """发送系统级通知"""
        try:
            system = platform.system()
            if system == "Windows":
                from win10toast import ToastNotifier
                ToastNotifier().show_toast("抖音下载器", message, duration=5)
            elif system == "Linux":
                os.system(f'notify-send "抖音下载器" "{message}" --icon=dialog-information')
            elif system == "Darwin":
                os.system(f'osascript -e \'display notification "{message}" with title "抖音下载器"\'')
        except Exception as e:
            print(f"系统通知发送失败: {str(e)}")

# ====================== 核心下载器 ======================
class DouyinDownloaderV4:
    def __init__(self, cookie: str, save_dir: str, max_workers: int = 3):
        self.base_headers = {
            'User-Agent': self._random_ua(),
            'Cookie': cookie,
            'Referer': 'https://www.douyin.com/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
        }
        self.api_params = {
            'aid': 6383,
            'channel': 'channel_pc_web',
            'device_platform': 'web',
            'pc_client_type': 1
        }
        self.save_dir = Path(folder_paths.get_output_directory()) / save_dir
        self.meta_dir = self.save_dir / "metadata"
        self.max_workers = max_workers
        self._prepare_dirs()

    def _prepare_dirs(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def _random_ua(self):
        chrome_version = f"{random.randint(90, 122)}.0.{random.randint(1000, 9999)}.{random.randint(10, 200)}"
        return f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36"

    def _process_image_url(self, url):
        """特殊处理图片URL"""
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        
        # 保留必要参数
        keep_params = ['x-expires', 'from', 's', 'se', 'sc', 'biz_tag', 'l']
        new_query = {k: v[0] for k, v in query.items() if k in keep_params}
        
        # 添加图片质量参数
        if 'aweme_images' in parsed.path:
            new_query['quality'] = '100'
        
        return urlunparse(parsed._replace(query=urlencode(new_query)))

    async def get_sec_uid(self, short_url: str):
        async with aiohttp.ClientSession(headers=self.base_headers) as session:
            async with session.get(short_url, allow_redirects=True) as resp:
                final_url = str(resp.url)
                if 'user/' not in final_url:
                    raise ValueError("无效的账号链接")
                return final_url.split('user/')[1].split('?')[0]

    async def fetch_all_aweme(self, sec_uid: str, max_count: int):
        aweme_list = []
        cursor = 0
        retry = 0
        with tqdm(desc="获取作品数据", unit="page") as pbar:
            while len(aweme_list) < max_count and retry < 5:
                params = {**self.api_params, 
                         'sec_user_id': sec_uid,
                         'count': 20,
                         'max_cursor': cursor}
                
                try:
                    await asyncio.sleep(random.uniform(2.0, 4.0))
                    async with aiohttp.ClientSession(headers=self.base_headers) as session:
                        async with session.get(
                            'https://www.douyin.com/aweme/v1/web/aweme/post/',
                            params=params
                        ) as response:
                            data = await response.json()
                            if data.get('status_code') != 0:
                                if data.get('status_code') == 8:
                                    raise PermissionError("Cookie无效或过期")
                                raise RuntimeError(f"接口错误: {data.get('status_msg')}")
                            
                            aweme_list.extend(data.get('aweme_list', []))
                            cursor = data.get('max_cursor', 0)
                            retry = 0
                            pbar.update(1)
                            
                            if data.get('has_more') == 0:
                                break
                except Exception as e:
                    print(f"获取作品失败: {str(e)}")
                    retry += 1
                    await asyncio.sleep(3 ** retry)
        
        return aweme_list[:max_count]

    async def _download_media(self, session, url, filepath, referer):
        for attempt in range(5):
            try:
                headers = {
                    **self.base_headers,
                    'Referer': referer,
                    'User-Agent': self._random_ua(),
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8' if url.endswith(('.jpg', '.webp')) else '*/*'
                }
                
                # 特殊处理图片URL
                final_url = self._process_image_url(url) if 'aweme_images' in url else url

                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(final_url, headers=headers, timeout=timeout) as resp:
                    if resp.status != 200:
                        continue
                    
                    # 检查实际内容类型
                    content_type = resp.headers.get('Content-Type', '')
                    if 'image' in content_type and not filepath.suffix == '.jpg':
                        filepath = filepath.with_suffix('.jpg')
                    
                    with open(filepath, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    # 验证文件完整性
                    if filepath.stat().st_size == 0:
                        filepath.unlink()
                        continue
                        
                    return True
            except Exception as e:
                print(f"下载失败（第{attempt+1}次尝试） {url}: {str(e)}")
                await asyncio.sleep(random.uniform(1, 3))
        return False

    async def download_media(self, aweme_list: list):
        semaphore = asyncio.Semaphore(self.max_workers)
        success_count = 0
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for aweme in aweme_list:
                aweme_id = aweme['aweme_id']
                create_time = aweme['create_time']
                desc = aweme.get('desc', '')[:40].strip()
                safe_desc = ''.join(c for c in desc if c.isalnum() or c in (' ', '_')).rstrip()
                base_name = f"{safe_desc}_{aweme_id}_{create_time}" if safe_desc else f"{aweme_id}_{create_time}"
                referer_url = f"https://www.douyin.com/video/{aweme_id}"

                # 保存元数据
                meta_path = self.meta_dir / f"{aweme_id}.json"
                if not meta_path.exists():
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'desc': desc,
                            'statistics': aweme.get('statistics', {}),
                            'author': aweme.get('author', {}),
                            'music': aweme.get('music', {}),
                            'aweme_type': aweme.get('aweme_type')
                        }, f, ensure_ascii=False, indent=2)

                # 视频下载
                if aweme.get('aweme_type') == 0:
                    video_url = aweme['video']['play_addr']['url_list'][0]
                    video_ext = 'mp4'
                    fname = self.save_dir / f"{base_name}.{video_ext}"
                    tasks.append(self._download_media(session, video_url, fname, referer_url))

                # 图片下载
                elif aweme.get('aweme_type') in [2, 68]:
                    for idx, image in enumerate(aweme.get('images', [])):
                        image_url = image['url_list'][0]
                        fname = self.save_dir / f"{base_name}_p{idx+1}.jpg"
                        tasks.append(self._download_media(session, image_url, fname, referer_url))

            # 执行下载并显示进度
            with tqdm(total=len(tasks), desc="下载进度") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    if result:
                        success_count += 1
                    pbar.update(1)
        
        return success_count

# ====================== ComfyUI节点 ======================
class DouyinDownloadNode:
    @classmethod
    def INPUT_TYPES(cls):
        tutorial_text = """【使用说明】（同上，略）"""
        return {
            "required": {
                "cookie": ("STRING", {"multiline": True, "default": "passport_csrf_token=..."}),
                "account_url": ("STRING", {"default": "https://v.douyin.com/xxxx"}),
                "save_directory": ("STRING", {"default": "douyin_downloads"}),
                "max_download": ("INT", {"default": 50, "min": 1, "max": 2000}),
                "concurrency": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
            "optional": {
                "debug_mode": ("BOOLEAN", {"default": False}),
                "tutorial": ("STRING", {"multiline": True, "default": tutorial_text}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "execute"
    CATEGORY = "🎨公众号懂AI的木子做号工具/抖音批量下载器"
    OUTPUT_NODE = True

    def __init__(self):
        self.is_running = False
        self.current_status = "🟡 等待启动"
        self.last_log = ""
        self.success_count = 0

    def execute(self, cookie, account_url, save_directory, max_download, concurrency, debug_mode=False, **kwargs):
        if self.is_running:
            return ("🔴 当前有任务正在运行",)
            
        self.is_running = True
        self.success_count = 0
        self._update_status("🟢 任务初始化...")
        
        threading.Thread(
            target=self._async_execute,
            args=(cookie, account_url, save_directory.strip(), max_download, concurrency, debug_mode),
            daemon=True
        ).start()
        
        return (self.current_status,)

    def _async_execute(self, cookie, account_url, save_directory, max_download, concurrency, debug_mode):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            self._update_status("🟠 正在初始化下载器...")
            downloader = DouyinDownloaderV4(
                cookie=cookie,
                save_dir=save_directory,
                max_workers=concurrency
            )
            if debug_mode:
                print(f"[DEBUG] 保存目录: {downloader.save_dir}")

            self._update_status("🟠 解析用户信息...")
            sec_uid = loop.run_until_complete(downloader.get_sec_uid(account_url))
            if debug_mode:
                print(f"[DEBUG] 获取到SecUID: {sec_uid}")

            self._update_status(f"🟠 获取前{max_download}个作品...")
            aweme_list = loop.run_until_complete(downloader.fetch_all_aweme(sec_uid, max_download))
            if debug_mode:
                print(f"[DEBUG] 获取到{len(aweme_list)}个作品")

            self._update_status(f"🔵 开始下载{len(aweme_list)}个作品...")
            start_time = time.time()
            success_count = loop.run_until_complete(downloader.download_media(aweme_list))
            
            time_cost = time.time() - start_time
            self.success_count = success_count
            status_msg = (
                f"✅ 下载完成！成功{success_count}/{len(aweme_list)}个作品\n"
                f"📁 保存路径: {downloader.save_dir}\n"
                f"⏱️ 耗时: {time_cost:.1f}秒"
            )
            self._update_status(status_msg)
            
            os.system('echo "\a"')
            Notifier.notify(f"下载完成: {success_count}个作品")

        except Exception as e:
            error_msg = f"❌ 错误: {str(e)}"
            self._update_status(error_msg)
            if debug_mode:
                import traceback
                traceback.print_exc()
        finally:
            self.is_running = False
            loop.close()

    def _update_status(self, message):
        self.current_status = message
        if message != self.last_log:
            print(f"[Douyin Downloader] {message}")
            self.last_log = message

NODE_CLASS_MAPPINGS = {
    "DouyinDownloadNode": DouyinDownloadNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DouyinDownloadNode": "🎸 抖音作品下载器（视频版）",
    
}