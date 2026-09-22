// 验收第二轮：在文件顶部插入这几行，让下面所有行号整体下移 6 行。
// 锚点哈希取的是目标行前后各 2 行的代码内容，与行号无关，
// 所以上一轮发在 equals 上的那条评论应当被跨提交去重跳过。
//
//
package com.example.session;

import java.util.*;
import java.io.IOException;

/** 用户会话管理器：验收夹具（小文件，走方法完整体分支）。 */
public class UserSessionManager {

    private String status;

    /** 是否管理员。 */
    public boolean isAdmin(String user) {
        if (user == "admin")
            return true;
        return false;
    }

    /** 安静地关闭连接。 */
    public void closeQuietly(java.sql.Connection conn) {
        try {
            conn.close();
        } catch (Exception e) {
        }
    }

    /** 标记状态。 */
    public void mark(String next) {
        this.status = next;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof UserSessionManager;
    }
}
