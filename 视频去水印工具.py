# -*- coding: utf-8 -*-
"""
视频去水印工具 Pro (LaMa AI修复版)
- AI 自动检测水印区域（边缘+对比度+半透明分析）
- LaMa 深度学习 AI 修复（非模糊填充，效果自然）
- 支持手动框选调整
- 视频逐帧处理 + 音频保留
"""

import sys

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os
import threading
import cv2
import numpy as np

# 懒加载 LaMa 模型
_inpainter = None
def get_inpainter():
    global _inpainter
    if _inpainter is None:
        from lama_inpaint import LaMaInpainter
        _inpainter = LaMaInpainter()
    return _inpainter

# ========== FFmpeg 路径 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_EXE = None
FFPROBE_EXE = None
for p in [os.path.join(BASE_DIR, "ffmpeg", "bin", "ffmpeg.exe"), r"D:\qsy\ffmpeg\bin\ffmpeg.exe"]:
    if os.path.exists(p):
        FFMPEG_EXE = p
        break
for p in [os.path.join(BASE_DIR, "ffmpeg", "bin", "ffprobe.exe"), r"D:\qsy\ffmpeg\bin\ffprobe.exe"]:
    if os.path.exists(p):
        FFPROBE_EXE = p
        break
if not FFMPEG_EXE:
    try:
        r = subprocess.run(["where", "ffmpeg"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            FFMPEG_EXE = r.stdout.strip().split("\n")[0].strip()
            FFPROBE_EXE = os.path.join(os.path.dirname(FFMPEG_EXE), "ffprobe.exe")
    except:
        pass


class Region:
    def __init__(self, x, y, w, h, auto=False):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.auto = auto
        self.rect_id = None
        self.label_id = None
        self.handles = []  # 8个控制点的canvas id
        self.selected = False
        self.keyframes = []  # 关键帧列表: [(time, x, y, w, h), ...]

    def add_keyframe(self, time):
        """在指定时间添加关键帧，记录当前位置"""
        # 移除同一时间的旧关键帧
        self.keyframes = [k for k in self.keyframes if abs(k[0] - time) > 0.05]
        self.keyframes.append((time, self.x, self.y, self.w, self.h))
        self.keyframes.sort(key=lambda k: k[0])

    def get_position(self, time):
        """根据时间插值计算位置，返回(x, y, w, h)"""
        if not self.keyframes:
            return self.x, self.y, self.w, self.h

        # 时间在第一个关键帧之前
        if time <= self.keyframes[0][0]:
            return self.keyframes[0][1], self.keyframes[0][2], self.keyframes[0][3], self.keyframes[0][4]

        # 时间在最后一个关键帧之后
        if time >= self.keyframes[-1][0]:
            return self.keyframes[-1][1], self.keyframes[-1][2], self.keyframes[-1][3], self.keyframes[-1][4]

        # 在两个关键帧之间线性插值
        for i in range(len(self.keyframes) - 1):
            t1, x1, y1, w1, h1 = self.keyframes[i]
            t2, x2, y2, w2, h2 = self.keyframes[i + 1]
            if t1 <= time <= t2:
                ratio = (time - t1) / (t2 - t1) if t2 != t1 else 0
                x = int(x1 + (x2 - x1) * ratio)
                y = int(y1 + (y2 - y1) * ratio)
                w = int(w1 + (w2 - w1) * ratio)
                h = int(h1 + (h2 - h1) * ratio)
                return x, y, w, h

        return self.x, self.y, self.w, self.h

    def is_dynamic(self):
        return len(self.keyframes) > 0


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("图片/视频去水印工具 (LaMa AI修复版)")
        self.root.geometry("1080x800")
        self.root.minsize(900, 700)
        self.root.configure(bg="#ececec")

        # 模式：'video' / 'image'
        self.mode = None

        # 视频
        self.video_path = None
        self.video_w = 0
        self.video_h = 0
        self.fps = 30
        self.duration = 0
        self.total_frames = 0
        self.cur_time = 0
        self.cap = None

        # 图片
        self.image_path = None
        self.image_bgr = None  # 原始图片BGR
        self.image_result = None  # 处理后的图片

        # 显示
        self.disp_w = 0
        self.disp_h = 0
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self.photo = None
        self.current_frame = None  # 当前帧的numpy数组（原始分辨率）

        # 选区
        self.regions = []
        self.drawing = False
        self.sx = self.sy = 0
        self.tmp_rect = None
        self.selected_region = None
        self.drag_mode = None  # 'move' / 'nw' / 'n' / 'ne' / 'e' / 'se' / 's' / 'sw' / 'w'
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.orig_region = None

        # 播放
        self.playing = False
        self._play_after_id = None
        self._seek_after_id = None

        self._build()

    def _build(self):
        # 顶部
        top = tk.Frame(self.root, bg="#ececec", pady=8)
        top.pack(fill=tk.X, padx=16)

        tk.Button(top, text="打开视频", command=self.open_video,
                  bg="#1677ff", fg="white", font=("微软雅黑", 10, "bold"),
                  relief=tk.FLAT, padx=22, pady=7, cursor="hand2").pack(side=tk.LEFT)
        tk.Button(top, text="打开图片", command=self.open_image,
                  bg="#13c2c2", fg="white", font=("微软雅黑", 10, "bold"),
                  relief=tk.FLAT, padx=22, pady=7, cursor="hand2").pack(side=tk.LEFT, padx=6)

        self.file_label = tk.Label(top, text="未打开文件", bg="#ececec",
                                    font=("微软雅黑", 9), fg="#666")
        self.file_label.pack(side=tk.LEFT, padx=12)

        tk.Button(top, text="AI 一键去水印", command=self.ai_one_click,
                  bg="#722ed1", fg="white", font=("微软雅黑", 9, "bold"),
                  relief=tk.FLAT, padx=16, pady=6, cursor="hand2").pack(side=tk.RIGHT, padx=4)
        tk.Button(top, text="清空全部", command=self.clear_all,
                  bg="#ff4d4f", fg="white", font=("微软雅黑", 9),
                  relief=tk.FLAT, padx=16, pady=6, cursor="hand2").pack(side=tk.RIGHT, padx=4)

        # 提示
        tip = tk.Label(self.root,
                       text="LaMa AI修复版：AI自动检测水印位置，用深度学习AI修复算法自然去除水印（非模糊填充），也可手动框选调整",
                       bg="#ececec", fg="#888", font=("微软雅黑", 9))
        tip.pack(fill=tk.X, padx=16, pady=(0, 4))

        # 视频区
        vf = tk.Frame(self.root, bg="#000", bd=0)
        vf.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 4))

        self.canvas = tk.Canvas(vf, bg="#000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_lbtn_down)
        self.canvas.bind("<B1-Motion>", self.on_lbtn_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_lbtn_up)
        self.canvas.bind("<ButtonPress-3>", self.on_rbtn_down)
        self.canvas.bind("<Configure>", self.on_resize)

        self.center_tip = tk.Label(vf, text="点击左上角「打开视频」",
                                    bg="#000", fg="#555", font=("微软雅黑", 14))
        self.center_tip.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # 控制条
        ctrl = tk.Frame(self.root, bg="#222", padx=12, pady=6)
        ctrl.pack(fill=tk.X, padx=16, pady=(0, 4))

        self.play_btn = tk.Button(ctrl, text="▶", command=self.toggle_play,
                                   bg="#333", fg="white", font=("Arial", 10),
                                   relief=tk.FLAT, padx=10, pady=3, cursor="hand2",
                                   state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT)

        self.time_lbl = tk.Label(ctrl, text="00:00 / 00:00", bg="#222",
                                  fg="#ccc", font=("Consolas", 10))
        self.time_lbl.pack(side=tk.LEFT, padx=10)

        self.seek_var = tk.DoubleVar(value=0)
        self.seek = ttk.Scale(ctrl, from_=0, to=100, orient=tk.HORIZONTAL,
                               variable=self.seek_var, command=self.on_seek,
                               state=tk.DISABLED)
        self.seek.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.info_lbl = tk.Label(ctrl, text="", bg="#222", fg="#888",
                                  font=("Consolas", 9))
        self.info_lbl.pack(side=tk.RIGHT, padx=8)

        # 选区信息
        rf = tk.Frame(self.root, bg="#fff", bd=1, relief=tk.SOLID)
        rf.pack(fill=tk.X, padx=16, pady=(0, 4))
        self.region_lbl = tk.Label(rf, text="水印区域：无  （手动框选或AI一键去水印，选中后可拖动调整）",
                                     bg="#fff", font=("微软雅黑", 9), fg="#333")
        self.region_lbl.pack(side=tk.LEFT, padx=10, pady=5)

        self.keyframe_btn = tk.Button(rf, text="➕ 添加关键帧", command=self.add_keyframe,
                                       bg="#fa8c16", fg="white", font=("微软雅黑", 9),
                                       relief=tk.FLAT, padx=12, pady=3, cursor="hand2",
                                       state=tk.DISABLED)
        self.keyframe_btn.pack(side=tk.RIGHT, padx=4)
        self.clear_kf_btn = tk.Button(rf, text="清除关键帧", command=self.clear_keyframes,
                                       bg="#8c8c8c", fg="white", font=("微软雅黑", 9),
                                       relief=tk.FLAT, padx=10, pady=3, cursor="hand2",
                                       state=tk.DISABLED)
        self.clear_kf_btn.pack(side=tk.RIGHT, padx=4)
        self.kf_info_lbl = tk.Label(rf, text="", bg="#fff", font=("微软雅黑", 9), fg="#fa8c16")
        self.kf_info_lbl.pack(side=tk.RIGHT, padx=8)

        # 底部
        bot = tk.Frame(self.root, bg="#ececec", pady=6)
        bot.pack(fill=tk.X, padx=16)

        self.out_lbl = tk.Label(bot, text="", bg="#ececec", font=("微软雅黑", 9), fg="#666")
        self.out_lbl.pack(side=tk.LEFT)

        self.go_btn = tk.Button(bot, text="AI 去水印", command=self.start,
                                 bg="#52c41a", fg="white", font=("微软雅黑", 11, "bold"),
                                 relief=tk.FLAT, padx=30, pady=8, cursor="hand2",
                                 state=tk.DISABLED)
        self.go_btn.pack(side=tk.RIGHT)

        # 处理进度
        pf = tk.Frame(self.root, bg="#ececec")
        pf.pack(fill=tk.X, padx=16, pady=(0, 10))
        self.prog = ttk.Progressbar(pf, mode="determinate", maximum=100)
        self.prog.pack(fill=tk.X, ipady=2)
        self.status_lbl = tk.Label(pf, text="就绪", bg="#ececec",
                                     font=("微软雅黑", 9), fg="#666")
        self.status_lbl.pack(pady=(3, 0))

    # ========== 打开视频 ==========
    def open_video(self):
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频", "*.mp4 *.mov *.avi *.mkv *.flv *.wmv *.webm *.m4v"),
                       ("所有", "*.*")])
        if not path:
            return

        if self.cap:
            self.cap.release()
            self.cap = None

        self.mode = "video"
        self.image_path = None
        self.image_bgr = None
        self.image_result = None
        self.video_path = path
        self.file_label.config(text=os.path.basename(path))
        self.clear_all()
        self.stop_play()
        self._load_info()

        if self.video_w > 0:
            self.cap = cv2.VideoCapture(path)
            self.seek.config(state=tk.NORMAL)
            self.play_btn.config(state=tk.NORMAL)
            self.center_tip.place_forget()
            self.go_btn.config(text="AI 去水印", command=self.start, state=tk.DISABLED)
            self.seek_to(0)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                       ("所有", "*.*")])
        if not path:
            return

        # 释放视频
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stop_play()

        # 读取图片
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("错误", "无法读取图片文件")
            return

        self.mode = "image"
        self.image_path = path
        self.image_bgr = img.copy()
        self.image_result = None
        self.video_w = img.shape[1]
        self.video_h = img.shape[0]
        self.current_frame = img.copy()

        self.file_label.config(text=os.path.basename(path))
        self.clear_all()

        # 禁用视频播放控制
        self.seek.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.time_lbl.config(text="图片模式")
        self.info_lbl.config(text=f"{self.video_w}×{self.video_h}  图片")
        self.center_tip.place_forget()

        # 显示图片
        self._show_image()

        # 更新按钮
        self.go_btn.config(text="AI 去水印", command=self.start, state=tk.DISABLED)
        self.status_lbl.config(text="图片已加载，框选水印区域或点「AI 一键去水印」")

    def _show_image(self):
        """显示图片到画布"""
        if self.image_bgr is None:
            return
        self.canvas.update()
        cw = max(self.canvas.winfo_width(), 80)
        ch = max(self.canvas.winfo_height(), 80)
        self.scale = min(cw / self.video_w, ch / self.video_h)
        self.disp_w = max(1, int(self.video_w * self.scale))
        self.disp_h = max(1, int(self.video_h * self.scale))

        frame_resized = cv2.resize(self.image_bgr, (self.disp_w, self.disp_h))
        _, buf = cv2.imencode(".png", frame_resized)
        self.photo = tk.PhotoImage(data=buf.tobytes())
        self._render()

    def _load_info(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            self.video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            cap.release()
            self.info_lbl.config(text=f"{self.video_w}×{self.video_h}  {self.fps:.1f}fps  {self.total_frames}帧")
        except Exception as e:
            messagebox.showerror("错误", f"读取视频信息失败：{e}")

    # ========== 帧显示 ==========
    def seek_to(self, t):
        if not self.video_path or self.video_w == 0 or not self.cap:
            return
        self.cur_time = max(0, min(t, self.duration))
        self._update_time()
        self.seek_var.set(self.cur_time / self.duration * 100 if self.duration > 0 else 0)

        # 用OpenCV读取帧
        frame_idx = int(self.cur_time * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame.copy()

        # 缩放显示
        self.canvas.update()
        cw = max(self.canvas.winfo_width(), 80)
        ch = max(self.canvas.winfo_height(), 80)
        self.scale = min(cw / self.video_w, ch / self.video_h)
        self.disp_w = max(1, int(self.video_w * self.scale))
        self.disp_h = max(1, int(self.video_h * self.scale))

        # 直接用BGR帧编码PNG（cv2.imencode期望BGR输入）
        frame_resized = cv2.resize(frame, (self.disp_w, self.disp_h))
        _, buf = cv2.imencode(".png", frame_resized)
        self.photo = tk.PhotoImage(data=buf.tobytes())
        self._render()

    def _render(self):
        if not self.photo:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.off_x = (cw - self.disp_w) // 2
        self.off_y = (ch - self.disp_h) // 2

        self.canvas.delete("all")
        self.canvas.create_image(self.off_x, self.off_y, anchor=tk.NW, image=self.photo)

        for i, r in enumerate(self.regions):
            self._draw_region(r, i + 1)

    def _draw_region(self, reg, idx):
        # 动态区域：正在拖动时用固定坐标显示，否则用插值
        if reg.is_dynamic() and not (self.selected_region is reg and self.drag_mode):
            rx, ry, rw, rh = reg.get_position(self.cur_time)
        else:
            rx, ry, rw, rh = reg.x, reg.y, reg.w, reg.h

        x1 = self.off_x + int(rx * self.scale)
        y1 = self.off_y + int(ry * self.scale)
        x2 = x1 + int(rw * self.scale)
        y2 = y1 + int(rh * self.scale)

        color = "#fa8c16" if reg.is_dynamic() else ("#722ed1" if reg.auto else "#ff4d4f")
        reg.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
        txt = f"动态{idx}" if reg.is_dynamic() else (f"AI{idx}" if reg.auto else str(idx))
        reg.label_id = self.canvas.create_text(x1 + 3, y1 + 3, anchor=tk.NW,
                                                 text=txt, fill="white",
                                                 font=("Arial", 9, "bold"))

        # 清除旧的控制点
        for h in reg.handles:
            self.canvas.delete(h)
        reg.handles = []

        # 选中时绘制8个控制点
        if reg.selected:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            handle_positions = [
                (x1, y1, "nw"), (cx, y1, "n"), (x2, y1, "ne"),
                (x2, cy, "e"), (x2, y2, "se"), (cx, y2, "s"),
                (x1, y2, "sw"), (x1, cy, "w")
            ]
            for hx, hy, _ in handle_positions:
                hid = self.canvas.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4,
                                                     fill="white", outline=color, width=1)
                reg.handles.append(hid)

    def on_resize(self, event):
        if self.mode == "image" and self.image_bgr is not None:
            self._show_image()
        elif self.video_path and self.video_w > 0:
            self.seek_to(self.cur_time)

    # ========== 播放 ==========
    def toggle_play(self):
        if self.playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self):
        if not self.video_path or not self.cap:
            return
        # 如果快到结尾了，从头开始
        if self.cur_time >= self.duration - 0.1:
            self.cur_time = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.playing = True
        self.play_btn.config(text="⏸")
        self._play_next()

    def stop_play(self):
        self.playing = False
        self.play_btn.config(text="▶")
        if self._play_after_id:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _play_next(self):
        if not self.playing or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_play()
            return
        self.current_frame = frame.copy()
        self.cur_time += 1.0 / max(self.fps, 1)
        if self.cur_time > self.duration:
            self.cur_time = self.duration
        self._update_time()
        self.seek_var.set(self.cur_time / self.duration * 100 if self.duration > 0 else 0)

        # 缩放显示
        cw = max(self.canvas.winfo_width(), 80)
        ch = max(self.canvas.winfo_height(), 80)
        self.scale = min(cw / self.video_w, ch / self.video_h)
        self.disp_w = max(1, int(self.video_w * self.scale))
        self.disp_h = max(1, int(self.video_h * self.scale))

        frame_resized = cv2.resize(frame, (self.disp_w, self.disp_h))
        _, buf = cv2.imencode(".png", frame_resized)
        self.photo = tk.PhotoImage(data=buf.tobytes())
        self._render()

        # 下一帧
        delay = int(1000 / max(self.fps, 1))
        self._play_after_id = self.root.after(delay, self._play_next)

    def on_seek(self, v):
        if self.duration > 0:
            # 拖动时停止播放
            if self.playing:
                self.stop_play()
            # 延迟更新，避免拖动时频繁跳转导致卡顿
            if self._seek_after_id:
                self.root.after_cancel(self._seek_after_id)
            t = float(v) / 100 * self.duration
            self.cur_time = max(0, min(t, self.duration))
            self._update_time()
            self._seek_after_id = self.root.after(50, lambda: self._do_seek(t))

    def _do_seek(self, t):
        self._seek_after_id = None
        self.seek_to(t)

    def _update_time(self):
        self.time_lbl.config(text=f"{self._fmt(self.cur_time)} / {self._fmt(self.duration)}")

    @staticmethod
    def _fmt(s):
        return f"{int(s//60):02d}:{int(s%60):02d}"

    # ========== 手动框选 ==========
    def _hit_test(self, x, y):
        """检测点击位置，返回(region, mode)，mode为'move'或控制点类型"""
        for r in reversed(self.regions):
            # 动态区域用插值后的位置
            if r.is_dynamic():
                rx, ry, rw, rh = r.get_position(self.cur_time)
            else:
                rx, ry, rw, rh = r.x, r.y, r.w, r.h

            x1 = self.off_x + int(rx * self.scale)
            y1 = self.off_y + int(ry * self.scale)
            x2 = x1 + int(rw * self.scale)
            y2 = y1 + int(rh * self.scale)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 检测控制点（8个）
            if r.selected:
                handles = [
                    (x1, y1, "nw"), (cx, y1, "n"), (x2, y1, "ne"),
                    (x2, cy, "e"), (x2, y2, "se"), (cx, y2, "s"),
                    (x1, y2, "sw"), (x1, cy, "w")
                ]
                for hx, hy, ht in handles:
                    if abs(x - hx) <= 5 and abs(y - hy) <= 5:
                        return r, ht

            # 检测区域内部
            if x1 <= x <= x2 and y1 <= y <= y2:
                return r, "move"

        return None, None

    def on_lbtn_down(self, event):
        if not self.photo:
            return

        # 先检测是否点击在已有区域上
        hit_reg, hit_mode = self._hit_test(event.x, event.y)

        if hit_reg:
            # 选中该区域，进入拖动/调整大小模式
            for r in self.regions:
                r.selected = False
            hit_reg.selected = True
            self.selected_region = hit_reg
            self.drag_mode = hit_mode
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            # 动态区域：把当前插值位置赋给固定坐标，方便拖动
            if hit_reg.is_dynamic():
                rx, ry, rw, rh = hit_reg.get_position(self.cur_time)
                hit_reg.x, hit_reg.y, hit_reg.w, hit_reg.h = rx, ry, rw, rh
            self.orig_region = (hit_reg.x, hit_reg.y, hit_reg.w, hit_reg.h)
            self._render()
            self._update_region_label()
            return

        # 点击在空白处，取消选中，开始绘制新区域
        for r in self.regions:
            r.selected = False
        self.selected_region = None
        self._render()
        self._update_region_label()

        self.drawing = True
        self.sx = event.x
        self.sy = event.y
        self.tmp_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                        outline="#fff", width=2, dash=(5, 3))

    def on_lbtn_drag(self, event):
        # 拖动/调整大小模式
        if self.selected_region and self.drag_mode:
            r = self.selected_region
            dx = (event.x - self.drag_start_x) / self.scale
            dy = (event.y - self.drag_start_y) / self.scale
            ox, oy, ow, oh = self.orig_region

            if self.drag_mode == "move":
                r.x = max(0, min(self.video_w - ow, ox + dx))
                r.y = max(0, min(self.video_h - oh, oy + dy))
            elif self.drag_mode == "nw":
                r.x = max(0, ox + dx)
                r.y = max(0, oy + dy)
                r.w = max(5, ow - (r.x - ox))
                r.h = max(5, oh - (r.y - oy))
            elif self.drag_mode == "n":
                r.y = max(0, oy + dy)
                r.h = max(5, oh - (r.y - oy))
            elif self.drag_mode == "ne":
                r.y = max(0, oy + dy)
                r.w = max(5, ow + dx)
                r.h = max(5, oh - (r.y - oy))
            elif self.drag_mode == "e":
                r.w = max(5, ow + dx)
            elif self.drag_mode == "se":
                r.w = max(5, ow + dx)
                r.h = max(5, oh + dy)
            elif self.drag_mode == "s":
                r.h = max(5, oh + dy)
            elif self.drag_mode == "sw":
                r.x = max(0, ox + dx)
                r.w = max(5, ow - (r.x - ox))
                r.h = max(5, oh + dy)
            elif self.drag_mode == "w":
                r.x = max(0, ox + dx)
                r.w = max(5, ow - (r.x - ox))

            # 限制在视频范围内
            r.x = max(0, min(self.video_w - 1, r.x))
            r.y = max(0, min(self.video_h - 1, r.y))
            r.w = min(self.video_w - r.x, r.w)
            r.h = min(self.video_h - r.y, r.h)

            self._render()
            return

        # 绘制新区域
        if not self.drawing or not self.tmp_rect:
            return
        self.canvas.coords(self.tmp_rect, self.sx, self.sy, event.x, event.y)

    def on_lbtn_up(self, event):
        # 结束拖动/调整大小
        if self.selected_region and self.drag_mode:
            r = self.selected_region
            # 动态区域：把拖动后的最终位置保存为当前时间的关键帧
            if r.is_dynamic():
                r.add_keyframe(self.cur_time)
                kf_count = len(r.keyframes)
                self.status_lbl.config(text=f"已更新 {self._fmt(self.cur_time)} 的关键帧（共{kf_count}个）")
            self.drag_mode = None
            self.selected_region = None
            self.orig_region = None
            self._update_region_label()
            self._render()
            return

        if not self.drawing:
            return
        self.drawing = False
        if self.tmp_rect:
            self.canvas.delete(self.tmp_rect)
            self.tmp_rect = None

        x1, y1 = min(self.sx, event.x), min(self.sy, event.y)
        x2, y2 = max(self.sx, event.x), max(self.sy, event.y)

        x1 = max(x1, self.off_x)
        y1 = max(y1, self.off_y)
        x2 = min(x2, self.off_x + self.disp_w)
        y2 = min(y2, self.off_y + self.disp_h)

        vx = max(0, int((x1 - self.off_x) / self.scale))
        vy = max(0, int((y1 - self.off_y) / self.scale))
        vx2 = min(self.video_w, int((x2 - self.off_x) / self.scale))
        vy2 = min(self.video_h, int((y2 - self.off_y) / self.scale))
        vw = vx2 - vx
        vh = vy2 - vy

        if vw > 2 and vh > 2:
            # 取消其他区域选中，选中新区域
            for r in self.regions:
                r.selected = False
            new_reg = Region(vx, vy, vw, vh, auto=False)
            new_reg.selected = True
            self.regions.append(new_reg)
            self.selected_region = new_reg
            self._render()
            self._update_region_label()
            self.go_btn.config(state=tk.NORMAL)

    def on_rbtn_down(self, event):
        if not self.photo:
            return
        for i in range(len(self.regions) - 1, -1, -1):
            r = self.regions[i]
            if r.is_dynamic():
                rx, ry, rw, rh = r.get_position(self.cur_time)
            else:
                rx, ry, rw, rh = r.x, r.y, r.w, r.h
            x1 = self.off_x + int(rx * self.scale)
            y1 = self.off_y + int(ry * self.scale)
            x2 = x1 + int(rw * self.scale)
            y2 = y1 + int(rh * self.scale)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                del self.regions[i]
                self._render()
                self._update_region_label()
                if not self.regions:
                    self.go_btn.config(state=tk.DISABLED)
                return

    def _update_region_label(self):
        if not self.regions:
            self.region_lbl.config(text="水印区域：无  （手动框选或AI一键去水印，选中后可拖动调整）")
            self.kf_info_lbl.config(text="")
            self.keyframe_btn.config(state=tk.DISABLED)
            self.clear_kf_btn.config(state=tk.DISABLED)
            return
        m = len([r for r in self.regions if not r.auto])
        a = len([r for r in self.regions if r.auto])
        d = len([r for r in self.regions if r.is_dynamic()])
        parts = []
        if m: parts.append(f"手动{m}个")
        if a: parts.append(f"AI{a}个")
        if d: parts.append(f"动态{d}个")
        self.region_lbl.config(text=f"水印区域：{' + '.join(parts)}（共{len(self.regions)}个），右键删除")

        # 更新关键帧按钮状态
        if self.selected_region:
            self.keyframe_btn.config(state=tk.NORMAL)
            if self.selected_region.is_dynamic():
                self.clear_kf_btn.config(state=tk.NORMAL)
                kf_count = len(self.selected_region.keyframes)
                self.kf_info_lbl.config(text=f"已标记 {kf_count} 个关键帧（动态）")
            else:
                self.clear_kf_btn.config(state=tk.DISABLED)
                self.kf_info_lbl.config(text="固定位置（添加关键帧可变为动态）")
        else:
            self.keyframe_btn.config(state=tk.DISABLED)
            self.clear_kf_btn.config(state=tk.DISABLED)
            self.kf_info_lbl.config(text="")

    def add_keyframe(self):
        """在当前时间点为选中区域添加关键帧"""
        if not self.selected_region:
            if self.regions:
                # 自动选中第一个区域
                for r in self.regions:
                    r.selected = False
                self.regions[0].selected = True
                self.selected_region = self.regions[0]
                self._render()
                self._update_region_label()
            else:
                messagebox.showinfo("提示", "请先框选一个水印区域")
                return
        self.selected_region.add_keyframe(self.cur_time)
        kf_count = len(self.selected_region.keyframes)
        self.status_lbl.config(text=f"已在 {self._fmt(self.cur_time)} 添加关键帧（共{kf_count}个）")
        self._update_region_label()
        self._render()

    def clear_keyframes(self):
        """清除选中区域的所有关键帧，变回固定位置"""
        if not self.selected_region:
            return
        self.selected_region.keyframes = []
        self.status_lbl.config(text="已清除关键帧，恢复为固定位置")
        self._update_region_label()
        self._render()

    def clear_all(self):
        self.regions = []
        self.selected_region = None
        self.drag_mode = None
        if self.canvas.winfo_exists():
            self.canvas.delete("all")
            if self.photo:
                self._render()
        self._update_region_label()
        # 重置按钮状态
        if self.mode == "image":
            self.go_btn.config(text="AI 去水印", command=self.start, state=tk.DISABLED)
        else:
            self.go_btn.config(state=tk.DISABLED)

    # ========== AI 一键去水印 ==========
    def ai_one_click(self):
        if self.current_frame is None:
            messagebox.showwarning("提示", "请先打开视频或图片")
            return

        self.status_lbl.config(text="AI 正在分析，自动检测水印并修复...")
        self.root.update()
        threading.Thread(target=self._ai_thread, daemon=True).start()

    def _ai_thread(self):
        try:
            frame = self.current_frame.copy()
            h, w = frame.shape[:2]

            # 转灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 方法1：Canny边缘检测 + 膨胀，找高密度边缘区域
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((5, 5), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)

            # 方法2：检测高亮/半透明文字区域（与背景对比度高）
            # 计算局部标准差，文字区域标准差高
            local_std = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 5)
            local_std = np.abs(gray.astype(np.float32) - local_std)
            local_std = (local_std / local_std.max() * 255).astype(np.uint8)

            # 合并边缘和高对比度区域
            combined = cv2.bitwise_or(edges_dilated, local_std)
            _, combined_thresh = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)

            # 进一步膨胀连通
            combined_thresh = cv2.dilate(combined_thresh, np.ones((8, 8), np.uint8), iterations=2)

            # 找连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                combined_thresh, connectivity=8)

            candidates = []
            for i in range(1, num_labels):
                x, y, cw, ch, area = stats[i]
                # 过滤：面积不能太小或太大
                area_ratio = area / (w * h)
                if 0.0005 < area_ratio < 0.15 and cw > 15 and ch > 8:
                    # 优先选择靠近四角的区域（水印通常在角落）
                    corner_score = self._corner_score(x, y, cw, ch, w, h)
                    candidates.append((x, y, cw, ch, area, corner_score))

            # 按角落优先级排序
            candidates.sort(key=lambda c: c[5], reverse=True)

            # 取前4个最可能的
            detected = []
            for (x, y, cw, ch, area, score) in candidates[:4]:
                # 扩展边界
                ex = max(0, x - 8)
                ey = max(0, y - 8)
                ew = min(w - ex, cw + 16)
                eh = min(h - ey, ch + 16)
                detected.append(Region(ex, ey, ew, eh, auto=True))

            # 如果全图检测没找到，专门检测四角
            if not detected:
                detected = self._detect_corners(frame, gray, w, h)

            self.root.after(0, lambda: self._ai_result(detected))

        except Exception as e:
            self.root.after(0, lambda: self.status_lbl.config(text=f"AI 检测出错：{e}"))

    @staticmethod
    def _corner_score(x, y, w, h, img_w, img_h):
        """计算区域靠近四角的程度，越高越可能是水印"""
        cx = x + w / 2
        cy = y + h / 2
        # 四个角的距离
        d1 = np.sqrt(cx**2 + cy**2)  # 左上
        d2 = np.sqrt((img_w - cx)**2 + cy**2)  # 右上
        d3 = np.sqrt(cx**2 + (img_h - cy)**2)  # 左下
        d4 = np.sqrt((img_w - cx)**2 + (img_h - cy)**2)  # 右下
        min_d = min(d1, d2, d3, d4)
        max_d = np.sqrt(img_w**2 + img_h**2)
        return 1.0 - min_d / max_d  # 越靠近角分数越高

    def _detect_corners(self, frame, gray, w, h):
        """专门检测四角的水印"""
        results = []
        cw = min(w // 3, 350)
        ch = min(h // 4, 200)

        corners = [
            (0, 0, cw, ch),
            (w - cw, 0, cw, ch),
            (0, h - ch, cw, ch),
            (w - cw, h - ch, cw, ch),
        ]

        for (cx, cy, cw_, ch_) in corners:
            roi = gray[cy:cy+ch_, cx:cx+cw_]
            if roi.size == 0:
                continue

            # 检测高亮像素（白色文字）
            _, bright = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
            bright_pixels = cv2.countNonZero(bright)
            total = roi.size

            if total > 0 and bright_pixels / total > 0.015:
                # 找亮像素的边界
                ys, xs = np.where(bright > 0)
                if len(xs) > 10:
                    min_x, max_x = xs.min(), xs.max()
                    min_y, max_y = ys.min(), ys.max()
                    rw = max_x - min_x
                    rh = max_y - min_y
                    if rw > 12 and rh > 6:
                        ex = max(0, cx + min_x - 10)
                        ey = max(0, cy + min_y - 10)
                        ew = min(w - ex, rw + 20)
                        eh = min(h - ey, rh + 20)
                        results.append(Region(ex, ey, ew, eh, auto=True))

        return results[:4]

    def _ai_result(self, detected):
        if not detected:
            self.status_lbl.config(text="AI 未检测到明显水印，请手动框选")
            tip = "未检测到明显水印。\n\n建议：\n直接在画面上拖动左键手动框选水印区域，然后点「AI 去水印」"
            messagebox.showinfo("AI 一键去水印", tip)
            return

        # 移除旧的AI检测结果
        self.regions = [r for r in self.regions if not r.auto]
        self.regions.extend(detected)
        self._render()
        self._update_region_label()
        self.go_btn.config(state=tk.NORMAL)
        self.status_lbl.config(text=f"AI 检测到 {len(detected)} 个区域，正在开始修复...")
        # 直接开始AI修复
        self.root.after(500, self.start)

    # ========== AI 去水印处理 ==========
    def start(self):
        if not self.regions:
            messagebox.showwarning("提示", "请先框选或用AI检测至少一个水印区域")
            return

        if self.mode == "image":
            # 图片模式：处理单张图片
            if self.image_bgr is None:
                return
            self.go_btn.config(state=tk.DISABLED, text="处理中...")
            self.prog["value"] = 30
            self.status_lbl.config(text="AI 正在修复图片...")
            self.root.update()
            threading.Thread(target=self._process_image, daemon=True).start()
            return

        # 视频模式
        if not self.video_path:
            return
        filename = os.path.basename(self.video_path)
        base, ext = os.path.splitext(filename)
        out = os.path.join(r"D:\qsy\scsp", f"{base}_AI去水印{ext}")
        self.out_lbl.config(text=f"输出：{out}")

        self.go_btn.config(state=tk.DISABLED, text="处理中...")
        self.prog["value"] = 0
        self.status_lbl.config(text="AI 正在逐帧修复...")
        self.stop_play()

        threading.Thread(target=self._process, args=(out,), daemon=True).start()

    def _process_image(self):
        """处理图片去水印"""
        try:
            # 创建mask
            mask = np.zeros((self.video_h, self.video_w), dtype=np.uint8)
            for r in self.regions:
                x1 = max(0, r.x)
                y1 = max(0, r.y)
                x2 = min(self.video_w, r.x + r.w)
                y2 = min(self.video_h, r.y + r.h)
                mask[y1:y2, x1:x2] = 255

            # 加载模型并修复
            from lama_inpaint import get_inpainter
            inpainter = get_inpainter()
            result = inpainter.inpaint(self.image_bgr, mask)

            self.image_result = result
            self.prog["value"] = 100

            # 显示结果
            self.root.after(0, self._show_image_result)

        except Exception as e:
            self.root.after(0, lambda: self._fail(f"图片处理出错：{e}"))

    def _show_image_result(self):
        """显示图片处理结果"""
        if self.image_result is None:
            return
        self.image_bgr = self.image_result.copy()
        self.current_frame = self.image_result.copy()
        self._show_image()
        self.status_lbl.config(text="图片修复完成！点「保存图片」保存")
        self.go_btn.config(state=tk.NORMAL, text="💾 保存图片", command=self.save_image)
        self.out_lbl.config(text="处理完成，点击「保存图片」保存")

    def save_image(self):
        """保存处理后的图片"""
        if self.image_result is None:
            return
        base, ext = os.path.splitext(os.path.basename(self.image_path))
        default_name = f"{base}_去水印.png"
        path = filedialog.asksaveasfilename(
            title="保存图片",
            defaultextension=".png",
            initialfile=default_name,
            initialdir=r"D:\qsy\scsp",
            filetypes=[("PNG图片", "*.png"), ("JPEG图片", "*.jpg"), ("所有", "*.*")])
        if not path:
            return
        # 保存图片（支持中文路径）
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imencode('.jpg', self.image_result, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(path)
        else:
            cv2.imencode('.png', self.image_result)[1].tofile(path)
        self.status_lbl.config(text=f"图片已保存：{path}")
        messagebox.showinfo("保存成功", f"图片已保存到：\n{path}")

    def _process(self, out):
        try:
            # 降分辨率处理：缩小一半处理，速度提升约4倍
            downscale = 0.5
            small_w = max(1, int(self.video_w * downscale))
            small_h = max(1, int(self.video_h * downscale))

            # 检查是否有动态区域
            has_dynamic = any(r.is_dynamic() for r in self.regions)

            # 创建固定mask（如果没有动态区域）
            fixed_mask = None
            fixed_mask_small = None
            if not has_dynamic:
                fixed_mask = np.zeros((self.video_h, self.video_w), dtype=np.uint8)
                for r in self.regions:
                    x1 = max(0, r.x)
                    y1 = max(0, r.y)
                    x2 = min(self.video_w, r.x + r.w)
                    y2 = min(self.video_h, r.y + r.h)
                    fixed_mask[y1:y2, x1:x2] = 255
                # 缩小版mask
                fixed_mask_small = cv2.resize(fixed_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

            # 加载 LaMa AI 模型
            self.root.after(0, lambda: self.status_lbl.config(text="正在加载 LaMa AI 模型..."))
            inpainter = get_inpainter()

            # 临时无声视频
            tmp_video = os.path.join(BASE_DIR, "_tmp_noaudio.mp4")

            # 逐帧 LaMa AI 修复
            cap = cv2.VideoCapture(self.video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(tmp_video, fourcc, self.fps, (self.video_w, self.video_h))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 计算当前帧的时间
                cur_time = frame_idx / self.fps if self.fps > 0 else 0

                if has_dynamic:
                    # 动态区域：逐帧重新计算mask
                    mask = np.zeros((self.video_h, self.video_w), dtype=np.uint8)
                    for r in self.regions:
                        if r.is_dynamic():
                            rx, ry, rw, rh = r.get_position(cur_time)
                        else:
                            rx, ry, rw, rh = r.x, r.y, r.w, r.h
                        x1 = max(0, rx)
                        y1 = max(0, ry)
                        x2 = min(self.video_w, rx + rw)
                        y2 = min(self.video_h, ry + rh)
                        mask[y1:y2, x1:x2] = 255
                    mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = fixed_mask
                    mask_small = fixed_mask_small

                # 降分辨率处理：缩小帧和mask
                frame_small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

                # LaMa AI 修复缩小后的图
                restored_small = inpainter.inpaint(frame_small, mask_small)

                # 放大回原尺寸
                restored_full = cv2.resize(restored_small, (self.video_w, self.video_h), interpolation=cv2.INTER_LINEAR)

                # 只替换mask区域，非水印区域保持原帧不变（保证全图画质）
                result = frame.copy()
                mask_3ch = (mask > 127)[:, :, None].repeat(3, axis=2)
                result[mask_3ch] = restored_full[mask_3ch]

                writer.write(result)

                frame_idx += 1
                if frame_idx % 5 == 0:
                    pct = min(99, frame_idx / self.total_frames * 100)
                    self.root.after(0, lambda p=pct, f=frame_idx: self._set_prog(p, f))

            cap.release()
            writer.release()

            self.root.after(0, lambda: self.status_lbl.config(text="正在合并音频..."))

            # 用FFmpeg合并音频
            cmd = [FFMPEG_EXE, "-y", "-i", tmp_video, "-i", self.video_path,
                   "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                   "-c:a", "aac", "-b:a", "192k",
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-shortest", out]
            subprocess.run(cmd, capture_output=True, timeout=600)

            # 清理临时文件
            if os.path.exists(tmp_video):
                os.remove(tmp_video)

            if os.path.exists(out):
                self.root.after(0, lambda: self._done(out))
            else:
                self.root.after(0, lambda: self._fail("处理失败：输出文件未生成"))

        except Exception as e:
            self.root.after(0, lambda: self._fail(f"处理出错：{e}"))

    def _set_prog(self, p, frame):
        self.prog["value"] = p
        self.status_lbl.config(text=f"AI 修复中... {int(p)}%（第{frame}/{self.total_frames}帧）")

    def _done(self, path):
        self.prog["value"] = 100
        self.status_lbl.config(text=f"完成！已保存：{path}")
        self.go_btn.config(state=tk.NORMAL, text="AI 去水印")
        messagebox.showinfo("完成", f"AI 去水印完成！\n\n共处理 {len(self.regions)} 个区域\n\n输出：{path}")

    def _fail(self, msg):
        self.status_lbl.config(text=msg)
        if self.mode == "image":
            self.go_btn.config(state=tk.NORMAL, text="AI 去水印", command=self.start)
        else:
            self.go_btn.config(state=tk.NORMAL, text="AI 去水印")
        messagebox.showerror("错误", msg)


# ========== 启动 ==========
def main():
    if not FFMPEG_EXE or not os.path.exists(FFMPEG_EXE):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", "未找到 FFmpeg！\n\n请确认 D:\\qsy\\ffmpeg\\bin\\ffmpeg.exe 存在")
        return

    root = tk.Tk()
    try:
        root.state("zoomed")
    except:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
