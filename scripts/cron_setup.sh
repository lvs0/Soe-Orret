#!/bin/bash
# SOE Cron Setup - Installe les tâches cron pour les agents

echo "Installing SOE cron jobs..."

# System Monitor (toutes les 15 min)
(crontab -l 2>/dev/null | grep -v "soe.*monitor"; echo "*/15 * * * * cd /home/l-vs/soe && python3 agents/monitor_agent.py >> logs/cron.log 2>&1") | crontab -

# Ruche Collector (quotidien à 2h du mat)
(crontab -l 2>/dev/null | grep -v "soe.*ruche"; echo "0 2 * * * cd /home/l-vs/soe && python3 agents/ruche_agent.py >> logs/cron.log 2>&1") | crontab -

# Loop Processor (après Ruche, 3h du mat)
(crontab -l 2>/dev/null | grep -v "soe.*processor"; echo "0 3 * * * cd /home/l-vs/soe && python3 agents/processor_agent.py >> logs/cron.log 2>&1") | crontab -

echo "Done. Current crontab:"
crontab -l | grep -E "soe|CLAW" || echo "(no SOE jobs)"

echo ""
echo "To remove: crontab -l | grep -v 'soe' | crontab -"