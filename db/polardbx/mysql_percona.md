# Ideia

Use polarx-rpc inside Percona MySQL 8 without distributed feature of PolarDBX Engine

Percona MySQL 8.0.39

Copied polarbx-engine/plugin/polarx_rpc/ directory to percona-server/plugin
In percona-server/build
make

# Adjustments

### Error
```bash
In file included from /data/percona-server/extra/protobuf/protobuf-24.4/src/google/protobuf/io/coded_stream.h:130,
 from /data/percona-server/build/plugin/polarx_rpc/protobuf_lite/polarx.pb.h:24,
 from /data/percona-server/build/plugin/polarx_rpc/protobuf_lite/polarx.pb.cc:4:
/data/percona-server/extra/protobuf/protobuf-24.4/src/google/protobuf/stubs/common.h:44:10: fatal error: absl/strings/string_view.h: Arquivo ou diretório inexistente
 44 | #include "absl/strings/string_view.h"
 | ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
make[2]:  [plugin/polarx_rpc/CMakeFiles/polarx_rpc_protobuf_objlib.dir/build.make:208: plugin/polarx_rpc/CMakeFiles/polarx_rpc_protobuf_objlib.dir/protobuf_lite/polarx.pb.cc.o] Erro 1
```

Install Abseil libraries used in Protobuf 24.4

```bash
git clone https://github.com/abseil/abseil-cpp.git
cd abseil-cpp
# Checkout a compatible version - LTS 20230802.1 should work with Protobuf 24.4
git checkout 20230802.1

mkdir build && cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      ..
make
sudo make install
```

### Error

```bash
[ 38%] Building CXX object plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/polarx_rpc.cc.o
In file included from /data/percona-server/plugin/polarx_rpc/polarx_rpc.cc:40:
/data/percona-server/plugin/polarx_rpc/server/server.h:39:10: fatal error: sql/sys_vars_ext.h: Arquivo ou diretório inexistente
 39 | #include "sql/sys_vars_ext.h"
 | ^~~~~~~~~~~~~~~~~~~~
compilation terminated.
make[2]:  [plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/build.make:76: plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/polarx_rpc.cc.o] Erro 1
make[1]:  [CMakeFiles/Makefile2:14639: plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/all] 
```

First I copied sys_vars_ext.h and sys_vars.ext.cc because it has a lot of dependencies.
I figured out that it only need rpc port configuration

### Error

```bash
In file included from /data/percona-server/plugin/polarx_rpc/server/listener.h:56,
 from /data/percona-server/plugin/polarx_rpc/server/server.h:55,
 from /data/percona-server/plugin/polarx_rpc/polarx_rpc.cc:40:
/data/percona-server/plugin/polarx_rpc/server/tcp_connection.h: In member function ‘void polarx_rpc::CtcpConnection::fin(const char*)’:
/data/percona-server/plugin/polarx_rpc/server/tcp_connection.h:184:13: error: ‘unireg_abort’ was not declared in this scope
 184 | unireg_abort(MYSQLD_ABORT_EXIT);
```

It seems that Polar uses unireg_abort function but it was removed in mysql 8.
Comparing mysql.h and sql/mysqld.c and sql/mysqld.h they introduced this function
I copied back the definition from sql/mysqld.h and removed static in mysqld.cc


### Error

```bash
[ 35%] Building CXX object plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/polarx_rpc.cc.o
In file included from ../../../plugin/polarx_rpc/server/listener.h:56,
                 from ../../../plugin/polarx_rpc/server/server.h:55,
                 from ../../../plugin/polarx_rpc/polarx_rpc.cc:40:
../../../plugin/polarx_rpc/server/tcp_connection.h: In member function ‘virtual bool polarx_rpc::CtcpConnection::events(uint32_t, int, int)’:
../../../plugin/polarx_rpc/server/tcp_connection.h:845:46: error: ‘ER_POLARX_RPC_ERROR_MSG’ was not declared in this scope; did you mean ‘ER_GRP_RPL_ERROR_MSG’?
  845 |                     PolarXRPC::Error::FATAL, ER_POLARX_RPC_ERROR_MSG,
      |                                              ^~~~~~~~~~~~~~~~~~~~~~~
      |                                              ER_GRP_RPL_ERROR_MSG
At global scope:
```

This error definition is generated in build/include/mysqld_error.h
The base file for this error is in share/messages_to_clients.txt
I included all POLAR and X_?? errors there and rebuild mysql with make

### Error

```bash
./../../plugin/polarx_rpc/session/session.cc:106:22: error: ‘class THD’ has no member named ‘polarx_rpc_enter’
 106 | auto before = thd->polarx_rpc_enter.fetch_add(1, std::memory_order_acquire);
 | ^~~~~~~~~~~~~~~~
../../../plugin/polarx_rpc/session/session.cc:121:12: error: ‘class THD’ has no member named ‘polarx_rpc_record’
 121 | thd->polarx_rpc_record = false;
 | ^~~~~~~~~~~~~~~~~
```

I added new fiels in THD (Thread Handle Data) to track session in PolarDBX RPC  
File: sql/sql_class

### Error

```bash
[ 35%] Building CXX object plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/session/session.cc.o
../../../plugin/polarx_rpc/session/session.cc: In member function ‘void polarx_rpc::Csession::dispatch(polarx_rpc::msg_t&&, bool&)’:
../../../plugin/polarx_rpc/session/session.cc:329:40: error: ‘class Srv_session’ has no member named ‘set_savepoint’
  329 |               err_no = mysql_session_->set_savepoint(sp_name);
      |                                        ^~~~~~~~~~~~~
../../../plugin/polarx_rpc/session/session.cc:332:40: error: ‘class Srv_session’ has no member named ‘release_savepoint’
  332 |               err_no = mysql_session_->release_savepoint(sp_name);
      |                                        ^~~~~~~~~~~~~~~~~
../../../plugin/polarx_rpc/session/session.cc:335:40: error: ‘class Srv_session’ has no member named ‘rollback_savepoint’
  335 |               err_no = mysql_session_->rollback_savepoint(sp_name);
      |                                        ^~~~~~~~~~~~~~~~~~
../../../plugin/polarx_rpc/session/session.cc: In member function ‘polarx_rpc::err_t polarx_rpc::Csession::sql_stmt_execute(const PolarXRPC::Sql::StmtExecute&)’:
../../../plugin/polarx_rpc/session/session.cc:411:54: error: ‘class THD’ has no member named ‘reset_gcn_variables’
  411 |   if (!thd->in_active_multi_stmt_transaction()) thd->reset_gcn_variables();
      |                                                      ^~~~~~~~~~~~~~~~~~~
../../../plugin/polarx_rpc/session/session.cc:413:20: error: ‘struct System_variables’ has no member named ‘innodb_current_snapshot_gcn’
  413 |     thd->variables.innodb_current_snapshot_gcn = true;
```

Lizzard code, commented out all in session.cc and session_base.cc

### Error

```bash
[ 35%] Building CXX object plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/session/session_base.cc.o
[ 35%] Building CXX object plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/session/session_manager.cc.o
../../../plugin/polarx_rpc/session/session_manager.cc:38:10: fatal error: sql/timestamp_service.h: Arquivo ou diretório inexistente
   38 | #include "sql/timestamp_service.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
make[2]: *** [plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/build.make:132: plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/session/session_manager.cc.o] Erro 1
make[1]: *** [CMakeFiles/Makefile2:14922: plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/all] Erro 2
make: *** [Makefile:166: all] Erro 2
```

It seems to be used for generating sequential transactions.
I am not going to use for rpc communication
Commented out the reference and code in session_manager.cc

### Error

```bash
[ 35%] Building CXX object plugin/polarx_rpc/CMakeFiles/polarx_rpc.dir/executor/handler_api.cc.o
../../../plugin/polarx_rpc/executor/handler_api.cc: In function ‘int rpc_executor::handler_set_key_read_only(ExecTable*)’:
../../../plugin/polarx_rpc/executor/handler_api.cc:243:42: error: ‘virtual int handler::extra(ha_extra_function)’ is private within this context
  243 |   return exec_table->table()->file->extra(HA_EXTRA_KEYREAD);
      |          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~
In file included from ../../../sql/partition_element.h:28,
                 from ../../../sql/partition_info.h:33,
                 from ../../../sql/log_event.h:66,
                 from ../../../sql/binlog_reader.h:28,
                 from ../../../sql/binlog.h:50,
                 from ../../../plugin/polarx_rpc/executor/handler_api.cc:33:
```

Changed to public in: sql/handler.h line 5752 

### Error

```bash
./../../plugin/polarx_rpc/executor/physical_backfill.cc: In member function ‘polarx_rpc::err_t rpc_executor::Physical_backfill::check_file_existence(const PolarXRPC::PhysicalBackfill::GetFileInfoOperator&, PolarXRPC::PhysicalBackfill::GetFileInfoOperator&)’:
../../../plugin/polarx_rpc/executor/physical_backfill.cc:152:73: error: ‘O_SHARE’ was not declared in this scope
  152 |           if ((file_desc_info.file = my_open(full_file_name, O_RDONLY | O_SHARE,
      |                                                                         ^~~~~~~
```

Add definitions

```c
#ifndef O_SHARE
#define O_SHARE 0
#endif

#ifndef O_BINARY
#ifdef _O_BINARY        // Windows style
#define O_BINARY _O_BINARY
#else
#define O_BINARY 0      // Unix systems don't need O_BINARY
#endif
#endif
```
