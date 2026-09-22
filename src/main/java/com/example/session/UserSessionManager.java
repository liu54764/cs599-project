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
