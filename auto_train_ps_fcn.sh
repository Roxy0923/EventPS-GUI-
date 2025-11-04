#!/bin/bash
# PS-FCN 自动恢复训练脚本
# 当训练因内存问题崩溃时，自动从最新checkpoint重新启动

set -e

# 配置参数
PROJECT_DIR="/home/c303/cxz/EventPS（GUI）/EventPS"
VENV_DIR="$PROJECT_DIR/venv"
TRAIN_SCRIPT="python/ps_fcn_train.py"
EXECUTABLE="./target/release/event_ps_train"
LOG_DIR="$PROJECT_DIR/logs"
MAX_RESTARTS=50  # 最大重启次数，防止无限循环
RESTART_DELAY=10  # 重启前等待时间（秒）

# 创建日志目录
mkdir -p "$LOG_DIR"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/auto_train_main_${TIMESTAMP}.log"
RESTART_LOG="$LOG_DIR/auto_train_restarts_${TIMESTAMP}.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} [INFO] $1" | tee -a "$RESTART_LOG"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} [WARN] $1" | tee -a "$RESTART_LOG"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} [ERROR] $1" | tee -a "$RESTART_LOG"
}

log_section() {
    echo -e "\n${BLUE}========================================${NC}" | tee -a "$RESTART_LOG"
    echo -e "${BLUE}$1${NC}" | tee -a "$RESTART_LOG"
    echo -e "${BLUE}========================================${NC}\n" | tee -a "$RESTART_LOG"
}

# 获取最新checkpoint的迭代次数
get_latest_iteration() {
    cd "$PROJECT_DIR"
    local latest_checkpoint=$(ls -t data/models/ev_ps_fcn_*.bin 2>/dev/null | head -1)
    if [ -n "$latest_checkpoint" ]; then
        # 从文件名提取迭代次数 (ev_ps_fcn_003000.bin -> 3000)
        local iter=$(basename "$latest_checkpoint" | sed 's/ev_ps_fcn_//;s/.bin//')
        echo "$iter"
    else
        echo "0"
    fi
}

# 检查LibreDR服务是否运行
check_libredr() {
    if ! pgrep -f "libredr_server" > /dev/null; then
        log_warn "LibreDR服务器未运行，尝试启动..."
        cd /home/c303/cxz/LibreDR
        nohup ./libredr_server examples/scripts/config_server.ini > libredr_server.log 2>&1 &
        sleep 2
        
        if ! pgrep -f "libredr_worker" > /dev/null; then
            nohup ./libredr_worker examples/scripts/config_worker.ini > libredr_worker.log 2>&1 &
            sleep 2
        fi
        
        log_info "LibreDR服务已启动"
    else
        log_info "LibreDR服务运行正常"
    fi
}

# 清理GPU资源
cleanup_gpu() {
    log_info "清理GPU资源..."
    # 清理可能残留的进程
    pkill -f "event_ps_train" || true
    sleep 2
}

# 训练函数
run_training() {
    local attempt=$1
    local current_iter=$(get_latest_iteration)
    
    log_section "第 $attempt 次训练尝试"
    log_info "当前checkpoint迭代次数: $current_iter"
    
    # 检查是否已完成
    if [ "$current_iter" -ge 10000 ]; then
        log_section "✓ 训练已完成！"
        log_info "最终模型: data/models/ev_ps_fcn_$(printf "%06d" $current_iter).bin"
        return 0
    fi
    
    # 检查LibreDR服务
    check_libredr
    
    # 进入项目目录
    cd "$PROJECT_DIR"
    
    # 激活虚拟环境
    source "$VENV_DIR/bin/activate"
    
    # 创建本次训练的日志文件
    local current_log="$LOG_DIR/train_attempt_${attempt}_iter${current_iter}_$(date +%Y%m%d_%H%M%S).log"
    
    log_info "开始训练（从第 $current_iter 次迭代继续）"
    log_info "训练日志: $current_log"
    
    # 启动训练
    DISPLAY=:1 PYTHONUNBUFFERED=1 "$EXECUTABLE" --ps-fcn-train "$TRAIN_SCRIPT" \
        2>&1 | tee "$current_log"
    
    # 获取退出码
    local exit_code=${PIPESTATUS[0]}
    
    # 检查退出原因
    if [ $exit_code -eq 0 ]; then
        log_info "训练正常退出（可能已完成）"
        return 0
    else
        log_error "训练异常退出，退出码: $exit_code"
        
        # 分析崩溃原因
        if grep -q "free(): invalid pointer" "$current_log" 2>/dev/null; then
            log_warn "检测到内存释放错误 (free(): invalid pointer)"
        elif grep -q "Segmentation fault" "$current_log" 2>/dev/null; then
            log_warn "检测到段错误 (Segmentation fault)"
        elif grep -q "double free" "$current_log" 2>/dev/null; then
            log_warn "检测到重复释放错误 (double free)"
        else
            log_warn "未知崩溃原因"
        fi
        
        # 获取崩溃时的迭代次数
        local crash_iter=$(grep "iter [0-9]* loss" "$current_log" 2>/dev/null | tail -1 | grep -oP 'iter \K[0-9]+' || echo "unknown")
        log_info "崩溃时迭代次数: $crash_iter"
        
        return 1
    fi
}

# 主循环
main() {
    log_section "PS-FCN 自动恢复训练脚本"
    log_info "项目目录: $PROJECT_DIR"
    log_info "最大重启次数: $MAX_RESTARTS"
    log_info "主日志文件: $MAIN_LOG"
    log_info "重启日志文件: $RESTART_LOG"
    
    local attempt=1
    
    while [ $attempt -le $MAX_RESTARTS ]; do
        if run_training $attempt; then
            # 训练正常完成
            local final_iter=$(get_latest_iteration)
            if [ "$final_iter" -ge 10000 ]; then
                log_section "🎉 训练完成！"
                log_info "总尝试次数: $attempt"
                log_info "最终迭代次数: $final_iter"
                log_info "最终模型: data/models/ev_ps_fcn_$(printf "%06d" $final_iter).bin"
                exit 0
            fi
        fi
        
        # 训练崩溃，准备重启
        log_warn "准备在 $RESTART_DELAY 秒后重启训练..."
        log_info "已完成 $attempt / $MAX_RESTARTS 次尝试"
        
        # 清理资源
        cleanup_gpu
        
        # 等待
        sleep $RESTART_DELAY
        
        attempt=$((attempt + 1))
    done
    
    # 达到最大重启次数
    log_section "❌ 已达到最大重启次数"
    log_error "训练未能完成，已尝试 $MAX_RESTARTS 次"
    log_info "请检查日志文件: $LOG_DIR"
    exit 1
}

# 捕获Ctrl+C信号
trap 'log_warn "收到中断信号，正在清理..."; cleanup_gpu; exit 130' INT TERM

# 运行主程序
main | tee -a "$MAIN_LOG"



