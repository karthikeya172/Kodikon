#!/usr/bin/env python3
"""
Kodikon Face Tracking - Complete Execution Pipeline
Runs all tests and integration in one command
"""

import subprocess
import sys
import time
from pathlib import Path


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_banner(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_stage(stage_num: int, stage_name: str, total_stages: int):
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print(f"┌{'─'*78}┐")
    print(f"│ STAGE {stage_num}/{total_stages}: {stage_name.ljust(60)} │")
    print(f"└{'─'*78}┘")
    print(f"{Colors.ENDC}")


def run_command(cmd: list, description: str) -> int:
    """Run command and return exit code"""
    print(f"\n{Colors.BLUE}▶ {description}{Colors.ENDC}")
    print(f"{Colors.BLUE}  Command: {' '.join(cmd)}{Colors.ENDC}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n{Colors.GREEN}✅ {description} - PASSED ({elapsed:.1f}s){Colors.ENDC}")
            return 0
        else:
            print(f"\n{Colors.RED}❌ {description} - FAILED (exit code: {result.returncode}){Colors.ENDC}")
            return result.returncode
    except Exception as e:
        print(f"\n{Colors.RED}❌ {description} - ERROR: {e}{Colors.ENDC}")
        return 1


def main():
    """Execute complete pipeline"""
    
    print_banner("🚀 KODIKON FACE TRACKING - COMPLETE EXECUTION PIPELINE")
    
    print(f"{Colors.YELLOW}{Colors.BOLD}")
    print("This script will execute:")
    print("  1. Standalone unit tests (9 tests)")
    print("  2. Integration examples (6 scenarios)")
    print("  3. Mock data generation")
    print("  4. Integrated system execution (camera + tracking + search)")
    print(f"{Colors.ENDC}")
    
    total_stages = 4
    failed_stages = []
    stage_times = []
    
    # ========================================================================
    # STAGE 1: UNIT TESTS
    # ========================================================================
    
    print_stage(1, "STANDALONE UNIT TESTS", total_stages)
    
    start = time.time()
    exit_code = run_command(
        ["python", "tests/test_backtrack_search_standalone.py"],
        "Running 9 unit tests (buffer, tracking, search, performance, concurrency)"
    )
    stage_times.append(time.time() - start)
    
    if exit_code != 0:
        failed_stages.append("Unit Tests")
    
    # ========================================================================
    # STAGE 2: INTEGRATION EXAMPLES
    # ========================================================================
    
    print_stage(2, "INTEGRATION EXAMPLES", total_stages)
    
    start = time.time()
    exit_code = run_command(
        ["python", "tests/integration_examples.py"],
        "Running 6 real-world integration examples"
    )
    stage_times.append(time.time() - start)
    
    if exit_code != 0:
        failed_stages.append("Integration Examples")
    
    # ========================================================================
    # STAGE 3: MOCK DATA UTILITIES
    # ========================================================================
    
    print_stage(3, "MOCK DATA GENERATION", total_stages)
    
    start = time.time()
    exit_code = run_command(
        ["python", "tests/mock_data_generator.py"],
        "Demonstrating synthetic data generators"
    )
    stage_times.append(time.time() - start)
    
    if exit_code != 0:
        failed_stages.append("Mock Data Generation")
    
    # ========================================================================
    # STAGE 4: INTEGRATED SYSTEM
    # ========================================================================
    
    print_stage(4, "INTEGRATED SYSTEM EXECUTION", total_stages)
    
    start = time.time()
    exit_code = run_command(
        ["python", "run_integrated_system.py", "--frames", "200", "--people", "5", "--searches", "4"],
        "Running complete integrated system (200 frames, 5 people, 4 searches)"
    )
    stage_times.append(time.time() - start)
    
    if exit_code != 0:
        failed_stages.append("Integrated System")
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    
    print_banner("📊 COMPLETE PIPELINE EXECUTION REPORT")
    
    print(f"{Colors.BOLD}Execution Summary:{Colors.ENDC}")
    print(f"  Total stages: {total_stages}")
    print(f"  Total time: {sum(stage_times):.1f} seconds")
    print()
    
    stages_info = [
        ("Unit Tests", stage_times[0]),
        ("Integration Examples", stage_times[1]),
        ("Mock Data Generation", stage_times[2]),
        ("Integrated System", stage_times[3]),
    ]
    
    for i, (stage_name, stage_time) in enumerate(stages_info, 1):
        status = "❌ FAILED" if stage_name in failed_stages else "✅ PASSED"
        print(f"  {i}. {stage_name.ljust(30)} {status.rjust(15)} ({stage_time:.1f}s)")
    
    print()
    
    if not failed_stages:
        print(f"{Colors.GREEN}{Colors.BOLD}")
        print("╔" + "═"*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + "✅ ALL STAGES PASSED - SYSTEM FULLY OPERATIONAL".center(78) + "║")
        print("║" + " "*78 + "║")
        print("╚" + "═"*78 + "╝")
        print(f"{Colors.ENDC}")
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}Key Achievements:{Colors.ENDC}")
        print(f"  ✓ 9 unit tests passed")
        print(f"  ✓ 6 integration examples ran successfully")
        print(f"  ✓ Mock data generators demonstrated")
        print(f"  ✓ Integrated system executed end-to-end")
        print(f"  ✓ 200+ frames processed")
        print(f"  ✓ 5 people tracked")
        print(f"  ✓ 4 backtrack searches completed")
        print(f"  ✓ 200+ events logged")
        
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print(f"  1. Review test results above")
        print(f"  2. Check documentation in docs/ folder")
        print(f"  3. Review integration code in tests/")
        print(f"  4. Integrate components into main system")
        print(f"  5. Deploy to production")
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}Documentation:{Colors.ENDC}")
        print(f"  • QUICK_TEST_REFERENCE.md - Quick start guide")
        print(f"  • TEST_BACKTRACK_GUIDE.md - Detailed testing guide")
        print(f"  • FACE_TRACKING_CODE_PATCHES.md - Code snippets")
        print(f"  • IMPLEMENTATION_CHECKLIST.md - Integration steps")
        
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}")
        print("╔" + "═"*78 + "╗")
        print("║" + " "*78 + "║")
        print(f"║ {'❌ SOME STAGES FAILED - SEE DETAILS ABOVE'.ljust(78)} ║")
        print("║" + " "*78 + "║")
        print("╚" + "═"*78 + "╝")
        print(f"{Colors.ENDC}")
        
        print(f"\n{Colors.RED}Failed stages:{Colors.ENDC}")
        for stage in failed_stages:
            print(f"  • {stage}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
