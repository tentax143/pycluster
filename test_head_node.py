#!/usr/bin/env python3
"""
Test script for PyCluster head node startup
"""

import asyncio
import sys
import os

# Add pycluster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pycluster'))

from pycluster.node import HeadNode

async def test_head_node():
    """Test head node startup and functionality"""
    print("🚀 Testing PyCluster Head Node...")
    
    try:
        # Create head node
        head = HeadNode(
            cluster_name='test-cluster',
            host='0.0.0.0',
            scheduler_port=8786,
            dashboard_port=8787
        )
        
        print("✅ Head node created successfully")
        
        # Start head node with 1 local worker
        print("🔄 Starting head node...")
        result = await head.start(n_local_workers=1)
        
        if result['status'] == 'success':
            print("✅ Head node started successfully!")
            print(f"   Status: {result['status']}")
            print(f"   Scheduler: {result.get('scheduler_address', 'N/A')}")
            print(f"   Dashboard: {result.get('dashboard_url', 'N/A')}")
            
            # Get connection info
            conn_info = head.get_connection_info()
            print("\n📋 Connection Information:")
            print(f"   Cluster Name: {conn_info['cluster_name']}")
            print(f"   Scheduler Address: {conn_info['scheduler_address']}")
            print(f"   Dashboard URL: {conn_info['dashboard_url']}")
            print(f"   Host IP: {conn_info['host_ip']}")
            
            # Get cluster status
            status = head.get_cluster_status()
            print(f"\n📊 Cluster Status:")
            print(f"   Status: {status.get('status', 'unknown')}")
            print(f"   Workers: {len(status.get('workers', []))}")
            
            print("\n🎉 Head node test completed successfully!")
            print("\n💡 You can now:")
            print("   1. Open dashboard at:", conn_info['dashboard_url'])
            print("   2. Connect workers using:", conn_info['scheduler_address'])
            print("   3. Press Ctrl+C to stop the head node")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down head node...")
                head.shutdown()
                print("✅ Head node shutdown complete")
                
        else:
            print(f"❌ Head node failed to start: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing head node: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("PyCluster Head Node Test")
    print("=" * 50)
    
    success = asyncio.run(test_head_node())
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
