
// ProjectRobot.h : PROJECT_NAME ���ε{�����D�n���Y��
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�� PCH �]�t���ɮ׫e���]�t 'stdafx.h'"
#endif

#include "resource.h"		// �D�n�Ÿ�


// CProjectRobotApp: 
// �аѾ\��@�����O�� ProjectRobot.cpp
//

class CProjectRobotApp : public CWinApp
{
public:
	CProjectRobotApp();

// �мg
public:
	virtual BOOL InitInstance();

// �{���X��@

	DECLARE_MESSAGE_MAP()
};

extern CProjectRobotApp theApp;